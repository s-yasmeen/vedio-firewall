"""Privacy-amplification primitives for TAPF-MIN.

These mechanisms are empirical controls for research evaluation. They do not by
 themselves constitute a formal differential-privacy guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable
import numpy as np


def privacy_blanket_sample(
    probabilities: np.ndarray,
    blanket_probability: float,
    seed: int,
) -> np.ndarray:
    """Sample a task-only one-hot release with a uniform privacy blanket.

    For each record, the categorical distribution is
        (1-alpha) * p(task|x) + alpha * Uniform(K).
    Increasing alpha removes confidence/detail and moves the release toward a
    task-independent distribution. The random draw is fresh for each invocation.
    """
    P = np.asarray(probabilities, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] < 2:
        raise ValueError("probabilities must be a 2-D array with >=2 classes")
    if not np.isfinite(P).all() or np.any(P < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not 0.0 <= blanket_probability <= 1.0:
        raise ValueError("blanket_probability must be in [0,1]")
    row_sum = P.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("each probability row must have positive mass")
    P = P / row_sum
    k = P.shape[1]
    alpha = float(blanket_probability)
    Q = (1.0 - alpha) * P + alpha / k
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(k, p=q) for q in Q], dtype=int)
    return np.eye(k, dtype=np.float32)[draws]


def hard_task_release(probabilities: np.ndarray) -> np.ndarray:
    """Release only the predicted task class; confidence is discarded."""
    P = np.asarray(probabilities)
    if P.ndim != 2:
        raise ValueError("probabilities must be 2-D")
    idx = np.argmax(P, axis=1)
    return np.eye(P.shape[1], dtype=np.float32)[idx]


@dataclass
class ReleaseLedger:
    """Fail-closed exposure ledger for repeated biometric-derived releases.

    The ledger limits how many *new* releases a session/task may emit. Repeated
    requests after the budget is exhausted are blocked rather than providing
    additional statistically independent evidence that could be averaged by an
    attacker. This is an operational composition control, not a DP accountant.
    """

    max_new_releases_per_key: int = 1
    counts: Dict[Hashable, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.max_new_releases_per_key) < 1:
            raise ValueError("max_new_releases_per_key must be >= 1")
        self.max_new_releases_per_key = int(self.max_new_releases_per_key)

    def remaining(self, key: Hashable) -> int:
        used = int(self.counts.get(key, 0))
        return max(0, self.max_new_releases_per_key - used)

    def allow_new_release(self, key: Hashable) -> bool:
        if self.remaining(key) <= 0:
            return False
        self.counts[key] = int(self.counts.get(key, 0)) + 1
        return True

    def status(self, key: Hashable) -> dict:
        used = int(self.counts.get(key, 0))
        remaining = self.remaining(key)
        return {
            "used": used,
            "remaining": remaining,
            "max_new_releases": self.max_new_releases_per_key,
            "decision": "ALLOW_NEW_RELEASE" if remaining > 0 else "BLOCK_NEW_RELEASE",
        }
