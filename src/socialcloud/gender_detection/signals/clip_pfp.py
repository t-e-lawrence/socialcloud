import asyncio
import io
import logging

import torch
from PIL import Image

from socialcloud.gender_detection.config import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    CLIP_PROMPT_GENDER_MAP,
    CLIP_PROMPTS,
)
from socialcloud.gender_detection.signals import SignalOutput

logger = logging.getLogger(__name__)


_device = None


def _get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("CLIP device: %s", _device)
    return _device


def load_clip_model():
    """Load CLIP model and pre-compute text embeddings. Call once at startup."""
    import open_clip

    device = _get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    text_tokens = tokenizer(CLIP_PROMPTS).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return model, preprocess, text_features


def _run_image_inference(image_bytes: bytes, model, preprocess, text_features) -> SignalOutput:
    device = _get_device()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

    idx = probs.argmax().item()
    gender = CLIP_PROMPT_GENDER_MAP[idx]
    if gender == "none":
        return SignalOutput(vote="none", probability=None)
    return SignalOutput(vote=gender, probability=probs[idx].item())


def _run_text_inference(text: str, model, text_features) -> SignalOutput:
    """Classify text (name or bio) by encoding it and comparing against gender prompts."""
    import open_clip

    device = _get_device()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
        probs = (100.0 * features @ text_features.T).softmax(dim=-1)[0]

    # Only consider male/female (indices 0 and 1), ignore "none" prompt
    female_prob = probs[0].item()
    male_prob = probs[1].item()
    if female_prob > male_prob:
        return SignalOutput(vote="female", probability=female_prob / (female_prob + male_prob))
    elif male_prob > female_prob:
        return SignalOutput(vote="male", probability=male_prob / (female_prob + male_prob))
    return SignalOutput(vote="none", probability=None)


async def classify_pfp(image_bytes: bytes | None, model, preprocess, text_features) -> SignalOutput:
    if not image_bytes:
        return SignalOutput(vote="none", probability=None)
    try:
        return await asyncio.to_thread(_run_image_inference, image_bytes, model, preprocess, text_features)
    except Exception:
        logger.exception("CLIP pfp classification failed")
        return SignalOutput(vote="none", probability=None)


async def classify_text(text: str | None, model, text_features) -> SignalOutput:
    if not text or not text.strip():
        return SignalOutput(vote="none", probability=None)
    try:
        return await asyncio.to_thread(_run_text_inference, text, model, text_features)
    except Exception:
        logger.exception("CLIP text classification failed")
        return SignalOutput(vote="none", probability=None)
