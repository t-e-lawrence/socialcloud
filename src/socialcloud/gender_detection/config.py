import os

DB_PATH = os.environ.get("GENDER_DB_PATH", "data/gender_detection.db")

CLIP_MODEL_NAME = "ViT-H-14"
CLIP_PRETRAINED = "dfn5b"
CLIP_PROMPTS = [
    "profile picture typically used by a woman or girl",
    "profile picture typically used by a man or boy",
    "abstract image, logo, solid color, or image with no gender signal",
]
CLIP_PROMPT_GENDER_MAP = ["female", "male", "none"]
