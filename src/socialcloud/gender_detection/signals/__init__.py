from dataclasses import dataclass


@dataclass
class SignalOutput:
    vote: str | None  # "male", "female", or "none"
    probability: float | None  # 0.0-1.0 normalized, or None if vote is "none"
