"""
Analysis API server — photo matching via CLIP embeddings.
"""

import base64
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from socialcloud.analysis.embeddings import _get_db, encode_image_from_bytes, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Preloaded embeddings: {username: [(content_type, vector), ...]}
_all_embeddings: dict[str, list[tuple[str, np.ndarray]]] = {}


def _load_all_embeddings():
    global _all_embeddings
    conn = _get_db()
    rows = conn.execute("SELECT username, content_type, embedding FROM embeddings").fetchall()
    _all_embeddings = {}
    for r in rows:
        vec = np.frombuffer(r["embedding"], dtype=np.float32).copy()
        _all_embeddings.setdefault(r["username"], []).append((r["content_type"], vec))
    conn.close()
    logger.info("Loaded %d embeddings for %d users", len(rows), len(_all_embeddings))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading CLIP model...")
    load_model()
    logger.info("Loading embeddings from DB...")
    _load_all_embeddings()
    yield


app = FastAPI(title="Analysis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchRequest(BaseModel):
    image_base64: str
    top_k: int = 20


class MatchResult(BaseModel):
    username: str
    similarity: float
    matched_content: str


@app.post("/match", response_model=list[MatchResult])
async def match(req: MatchRequest):
    image_bytes = base64.b64decode(req.image_base64)
    query_vec = encode_image_from_bytes(image_bytes)

    # Cosine similarity against all embeddings, take max per user
    user_scores = {}
    for username, embeddings in _all_embeddings.items():
        best_sim = -1.0
        best_type = ""
        for content_type, vec in embeddings:
            sim = float(np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_type = content_type
        user_scores[username] = (best_sim, best_type)

    ranked = sorted(user_scores.items(), key=lambda x: x[1][0], reverse=True)[: req.top_k]
    return [MatchResult(username=u, similarity=round(s, 4), matched_content=t) for u, (s, t) in ranked]


@app.get("/health")
async def health():
    return {"status": "ok", "users": len(_all_embeddings), "embeddings": sum(len(v) for v in _all_embeddings.values())}
