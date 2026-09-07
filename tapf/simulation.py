"""Deterministic simulation adapter used only for CI and controller validation."""
import math
from .temporal_privacy import accumulated_identity_evidence


def evaluate_simulated(alpha: float):
    # Deliberately deterministic. These values are NOT experimental evidence.
    identity_risk = min(1.0, 0.92 * math.exp(-3.5 * alpha) + 0.035)
    task_utility = max(0.0, 0.96 - 0.42 * (alpha ** 1.8))
    frame_risks = [max(0.0, min(1.0, identity_risk * f))
                   for f in (0.92, 0.96, 1.00, 1.04, 1.08)] * 6
    temporal_risk = accumulated_identity_evidence(frame_risks)
    return identity_risk, task_utility, temporal_risk
