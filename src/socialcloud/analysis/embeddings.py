"""
CLIP embedding pipeline — encode all user content into 512-dim vectors,
store per-item, compute per-user cloud metrics and pairwise distances.
"""

import io
import logging
import os
import sqlite3
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import wasserstein_distance_nd

logger = logging.getLogger(__name__)

DB_PATH = Path(os.environ.get("CRAWLER_DB_PATH", "data/network.db"))
DOWNLOADS_DIR = Path(os.environ.get("DOWNLOADS_DIR", "data/downloads"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".webm"}


def _get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ── CLIP model ────────────────────────────────────────────────────

_model = None
_preprocess = None
_tokenizer = None


_device = None


def _get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("CLIP device: %s", _device)
    return _device


def load_model():
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return _model, _preprocess, _tokenizer

    import open_clip

    device = _get_device()
    logger.info("Loading CLIP model (ViT-H-14 dfn5b)...")
    _model, _, _preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="dfn5b")
    _model.eval().to(device)
    _tokenizer = open_clip.get_tokenizer("ViT-H-14")
    logger.info("CLIP model loaded on %s.", device)
    return _model, _preprocess, _tokenizer


def encode_image(image_path: Path) -> np.ndarray:
    model, preprocess, _ = load_model()
    device = _get_device()
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_input)
        features /= features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy()


def encode_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    model, preprocess, _ = load_model()
    device = _get_device()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_input)
        features /= features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy()


def encode_text(text: str) -> np.ndarray:
    model, _, tokenizer = load_model()
    device = _get_device()
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy()


def _extract_video_frame(video_path: Path, output_dir: Path) -> Path | None:
    """Extract first frame from video at full resolution, save to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / (video_path.stem + ".jpg")
    if frame_path.exists():
        return frame_path
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vframes",
                "1",
                "-q:v",
                "2",
                str(frame_path),
            ],
            capture_output=True,
            timeout=15,
        )
        return frame_path if frame_path.exists() else None
    except Exception:
        return None


# ── Storage ───────────────────────────────────────────────────────


def store_embedding(
    conn,
    username: str,
    content_type: str,
    embedding: np.ndarray,
    source_file: str | None = None,
    source_text: str | None = None,
):
    conn.execute(
        "INSERT INTO embeddings (username, content_type, source_file, source_text, embedding) VALUES (?, ?, ?, ?, ?)",
        (
            username,
            content_type,
            source_file,
            source_text,
            embedding.astype(np.float32).tobytes(),
        ),
    )


def get_embeddings(conn, username: str, content_type: str | None = None) -> np.ndarray:
    if content_type:
        rows = conn.execute(
            "SELECT embedding FROM embeddings WHERE username = ? AND content_type = ?",
            (username, content_type),
        ).fetchall()
    else:
        rows = conn.execute("SELECT embedding FROM embeddings WHERE username = ?", (username,)).fetchall()
    if not rows:
        return np.array([])
    return np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])


def get_all_usernames_with_embeddings(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT DISTINCT username FROM embeddings").fetchall()]


def already_encoded(conn, username: str, content_type: str, source: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM embeddings WHERE username = ? AND content_type = ? AND (source_file = ? OR source_text = ?)",
        (username, content_type, source, source),
    ).fetchone()
    return row is not None


# ── Encoding pipeline ────────────────────────────────────────────


def encode_all():
    """Encode all downloaded media + text into embeddings."""
    conn = _get_db()

    # 1. Images + videos from downloads
    if DOWNLOADS_DIR.exists():
        profile_dirs = sorted([d for d in DOWNLOADS_DIR.iterdir() if d.is_dir()])
        logger.info("Encoding media for %d profiles...", len(profile_dirs))

        total_images = 0
        total_videos = 0
        for profile_dir in profile_dirs:
            username = profile_dir.name
            for subdir, content_type in [
                ("posts", "post_image"),
                ("stories", "story_image"),
                ("highlights", "highlight_image"),
            ]:
                target = profile_dir / subdir
                if not target.exists():
                    continue
                for media_path in sorted(target.rglob("*")):
                    if not media_path.is_file():
                        continue
                    ext = media_path.suffix.lower()
                    rel = str(media_path.relative_to(DOWNLOADS_DIR))
                    if already_encoded(conn, username, content_type, rel):
                        continue

                    if ext in IMAGE_EXTENSIONS:
                        try:
                            vec = encode_image(media_path)
                            store_embedding(conn, username, content_type, vec, source_file=rel)
                            total_images += 1
                        except Exception:
                            logger.warning("  Failed to encode image %s", media_path)

                    elif ext in VIDEO_EXTENSIONS:
                        thumb_dir = profile_dir / "video_thumbnails"
                        frame = _extract_video_frame(media_path, thumb_dir)
                        if frame:
                            try:
                                vec = encode_image(frame)
                                store_embedding(conn, username, content_type, vec, source_file=rel)
                                total_videos += 1
                            except Exception:
                                logger.warning("  Failed to encode video frame %s", media_path)

                    total_done = total_images + total_videos
                    if total_done > 0 and total_done % 50 == 0:
                        conn.commit()
                        logger.info(
                            "  Encoded %d media (%d images, %d videos)...",
                            total_done,
                            total_images,
                            total_videos,
                        )

        conn.commit()
        logger.info(
            "Encoded %d images + %d videos = %d total.",
            total_images,
            total_videos,
            total_images + total_videos,
        )

    # 2. Captions from posts table
    posts = conn.execute(
        "SELECT owner, post_id, caption FROM posts WHERE caption IS NOT NULL AND caption != ''"
    ).fetchall()
    total_text = 0
    for post in posts:
        if already_encoded(conn, post["owner"], "caption_text", post["caption"][:200]):
            continue
        try:
            vec = encode_text(post["caption"])
            store_embedding(
                conn,
                post["owner"],
                "caption_text",
                vec,
                source_text=post["caption"][:500],
            )
            total_text += 1
        except Exception:
            logger.warning("  Failed to encode caption for %s", post["post_id"])
    conn.commit()
    logger.info("Encoded %d captions.", total_text)

    # 3. Bios from profiles table
    profiles = conn.execute(
        "SELECT username, biography FROM profiles WHERE biography IS NOT NULL AND biography != '' AND crawl_status = 'crawled'"
    ).fetchall()
    total_bios = 0
    for p in profiles:
        if already_encoded(conn, p["username"], "bio_text", p["biography"][:200]):
            continue
        try:
            vec = encode_text(p["biography"])
            store_embedding(conn, p["username"], "bio_text", vec, source_text=p["biography"][:500])
            total_bios += 1
        except Exception:
            logger.warning("  Failed to encode bio for %s", p["username"])
    conn.commit()
    logger.info("Encoded %d bios.", total_bios)

    total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    logger.info("Total embeddings in DB: %d", total)
    conn.close()


# ── Per-user metrics ─────────────────────────────────────────────


def compute_user_metrics(username: str, conn=None) -> dict | None:
    own_conn = conn is None
    if own_conn:
        conn = _get_db()

    all_vecs = get_embeddings(conn, username)
    if len(all_vecs) < 2:
        if own_conn:
            conn.close()
        return None

    image_vecs = get_embeddings(conn, username, "post_image")
    text_vecs = get_embeddings(conn, username, "caption_text")

    result = {
        "username": username,
        "n_embeddings": len(all_vecs),
        "total_variance": float(np.var(all_vecs, axis=0).mean()),
        "visual_variance": float(np.var(image_vecs, axis=0).mean()) if len(image_vecs) >= 2 else None,
        "textual_variance": float(np.var(text_vecs, axis=0).mean()) if len(text_vecs) >= 2 else None,
    }

    if own_conn:
        conn.close()
    return result


def compute_all_metrics():
    """Compute and print cloud metrics for all users with embeddings."""
    conn = _get_db()
    usernames = get_all_usernames_with_embeddings(conn)
    logger.info("Computing metrics for %d users...", len(usernames))

    metrics = []
    for username in usernames:
        m = compute_user_metrics(username, conn)
        if m:
            metrics.append(m)

    metrics.sort(key=lambda m: m["total_variance"])

    print(f"\n{'=' * 70}")
    print(f"Per-user cloud metrics ({len(metrics)} users with 2+ embeddings)")
    print(f"{'=' * 70}")

    print("\nMost FOCUSED (lowest variance):")
    for m in metrics[:10]:
        vis = f"vis={m['visual_variance']:.6f}" if m["visual_variance"] is not None else "vis=N/A"
        txt = f"txt={m['textual_variance']:.6f}" if m["textual_variance"] is not None else "txt=N/A"
        print(f"  {m['username']:30s} var={m['total_variance']:.6f}  {vis}  {txt}  n={m['n_embeddings']}")

    print("\nMost ECLECTIC (highest variance):")
    for m in metrics[-10:]:
        vis = f"vis={m['visual_variance']:.6f}" if m["visual_variance"] is not None else "vis=N/A"
        txt = f"txt={m['textual_variance']:.6f}" if m["textual_variance"] is not None else "txt=N/A"
        print(f"  {m['username']:30s} var={m['total_variance']:.6f}  {vis}  {txt}  n={m['n_embeddings']}")

    conn.close()
    return metrics


# ── Pairwise distances ───────────────────────────────────────────


def compute_pairwise_distances() -> dict[tuple[str, str], float]:
    """Compute Wasserstein distance between all pairs of users with embeddings."""
    conn = _get_db()
    usernames = get_all_usernames_with_embeddings(conn)

    # Load all embeddings into memory
    user_vecs = {}
    for u in usernames:
        vecs = get_embeddings(conn, u)
        if len(vecs) >= 2:
            user_vecs[u] = vecs
    conn.close()

    users = sorted(user_vecs.keys())
    n_pairs = len(users) * (len(users) - 1) // 2
    logger.info(
        "Computing Wasserstein distances for %d pairs (%d users)...",
        n_pairs,
        len(users),
    )

    distances = {}
    done = 0
    for a, b in combinations(users, 2):
        distances[(a, b)] = float(wasserstein_distance_nd(user_vecs[a], user_vecs[b]))
        done += 1
        if done % 1000 == 0:
            logger.info("  %d/%d pairs computed...", done, n_pairs)

    logger.info("Done. %d pairwise distances computed.", len(distances))

    # Print most similar and most different pairs
    sorted_pairs = sorted(distances.items(), key=lambda x: x[1])
    print(f"\n{'=' * 70}")
    print(f"Pairwise Wasserstein distances ({len(distances)} pairs)")
    print(f"{'=' * 70}")

    print("\nMost SIMILAR pairs:")
    for (a, b), d in sorted_pairs[:10]:
        print(f"  {a:25s} ↔ {b:25s}  dist={d:.4f}")

    print("\nMost DIFFERENT pairs:")
    for (a, b), d in sorted_pairs[-10:]:
        print(f"  {a:25s} ↔ {b:25s}  dist={d:.4f}")

    return distances
