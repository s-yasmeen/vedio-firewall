"""Sequence-level privacy accounting for TAPF-MIN."""
import math
from typing import Sequence


def accumulated_identity_evidence(frame_risks: Sequence[float]) -> float:
    """Aggregate weak frame-wise identity evidence across time.

    This is a research proxy, not a formal privacy guarantee. Publication experiments
    should additionally use a real sequence-level identity attacker.
    """
    if not frame_risks:
        return 0.0
    eps = 1e-9
    total = 0.0
    for risk in frame_risks:
        r = min(max(float(risk), eps), 1.0 - eps)
        total += -math.log(1.0 - r)
    return total / len(frame_risks)


def worst_window_risk(frame_risks: Sequence[float], window: int = 30) -> float:
    if not frame_risks:
        return 0.0
    window = max(1, int(window))
    values = []
    for start in range(0, len(frame_risks)):
        chunk = frame_risks[start:start + window]
        if chunk:
            values.append(accumulated_identity_evidence(chunk))
    return max(values) if values else 0.0
