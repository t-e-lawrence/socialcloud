import base64
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from socialcloud.gender_detection.db import (
    close_db,
    get_profile,
    get_resolved_profiles,
    init_db,
    upsert_profile,
)
from socialcloud.gender_detection.models import ClassifyRequest, ClassifyResponse, SignalResult
from socialcloud.gender_detection.signals.clip_pfp import (
    classify_pfp,
    classify_text,
    load_clip_model,
)
from socialcloud.gender_detection.voter import composite_vote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    logger.info("Loading CLIP model...")
    model, preprocess, text_features = load_clip_model()
    app.state.clip_model = model
    app.state.clip_preprocess = preprocess
    app.state.clip_text_features = text_features
    logger.info("CLIP model loaded.")

    yield

    await close_db()


app = FastAPI(title="Gender Detection API", lifespan=lifespan)


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    existing = await get_profile(req.username)
    if existing and existing["status"] == "resolved":
        return _profile_to_response(existing)

    image_bytes = base64.b64decode(req.pfp_base64) if req.pfp_base64 else None

    clip_res = await classify_pfp(
        image_bytes,
        app.state.clip_model,
        app.state.clip_preprocess,
        app.state.clip_text_features,
    )
    # Combine name + bio into one text input
    text_parts = [p for p in [req.display_name, req.bio] if p]
    combined_text = " — ".join(text_parts) if text_parts else None
    text_res = await classify_text(
        combined_text,
        app.state.clip_model,
        app.state.clip_text_features,
    )

    signals = {"clip_pfp": clip_res, "clip_text": text_res}
    gender, breakdown = composite_vote(signals)

    await upsert_profile(
        username=req.username,
        display_name=req.display_name,
        signals=breakdown,
        gender=gender,
        status="resolved",
    )

    return ClassifyResponse(
        username=req.username,
        gender=gender,
        confidence_breakdown={k: SignalResult(**v) for k, v in breakdown.items()},
        status="resolved",
    )


@app.get("/status/{username}", response_model=ClassifyResponse)
async def status(username: str):
    profile = await get_profile(username)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return _profile_to_response(profile)


@app.get("/results")
async def results(gender: str | None = Query(None), limit: int = Query(50)):
    profiles = await get_resolved_profiles(gender=gender, limit=limit)
    return {"results": [_profile_to_response(p) for p in profiles]}


@app.get("/queue")
async def queue():
    return {"queued_profiles": 0, "note": "CLIP-only mode, no queue needed"}


def _profile_to_response(profile: dict) -> ClassifyResponse:
    breakdown = {k: SignalResult(**v) for k, v in profile["signals"].items()}
    return ClassifyResponse(
        username=profile["username"],
        gender=profile["gender"],
        confidence_breakdown=breakdown,
        status=profile["status"],
    )
