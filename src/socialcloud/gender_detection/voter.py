from socialcloud.gender_detection.signals import SignalOutput

# pfp gets 75% weight, text (name+bio) gets 25%
SIGNAL_WEIGHTS = {
    "clip_pfp": 0.75,
    "clip_text": 0.25,
}


def composite_vote(signals: dict[str, SignalOutput | None]) -> tuple[str | None, dict]:
    """
    Returns (gender, breakdown_dict).
    gender: "male", "female", or "undetermined".
    """
    scores = {"male": 0.0, "female": 0.0}
    breakdown = {}

    for name, signal in signals.items():
        if signal is None:
            breakdown[name] = {"vote": None, "raw_prob": None}
            continue
        breakdown[name] = {"vote": signal.vote, "raw_prob": signal.probability}
        if signal.vote in scores and signal.probability is not None:
            weight = SIGNAL_WEIGHTS.get(name, 0.25)
            scores[signal.vote] += signal.probability * weight

    if scores["male"] == 0.0 and scores["female"] == 0.0:
        gender = "undetermined"
    elif scores["male"] == scores["female"]:
        gender = "undetermined"
    else:
        gender = max(scores, key=scores.get)

    return gender, breakdown
