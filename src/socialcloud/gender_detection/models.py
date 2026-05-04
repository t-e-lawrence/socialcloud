from pydantic import BaseModel


class ClassifyRequest(BaseModel):
    username: str
    display_name: str | None = None
    bio: str | None = None
    pfp_base64: str | None = None


class SignalResult(BaseModel):
    vote: str | None = None
    raw_prob: float | None = None


class ClassifyResponse(BaseModel):
    username: str
    gender: str | None
    confidence_breakdown: dict[str, SignalResult]
    status: str
