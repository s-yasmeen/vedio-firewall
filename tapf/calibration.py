"""Calibration helpers for conservative TAPF-MIN release evidence.

These utilities convert repeated empirical measurements into confidence bounds used by
the deployment gate. They do not create privacy guarantees beyond the sampled threat model.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float
    n: int


def bootstrap_interval(values: Iterable[float], confidence: float = 0.95,
                       resamples: int = 2000, seed: int = 42) -> ConfidenceInterval:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("At least one measurement is required")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Measurements must be finite")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0")
    if resamples < 100:
        raise ValueError("resamples must be >= 100")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(resamples, arr.size))
    means = arr[idx].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return ConfidenceInterval(float(arr.mean()), float(lower), float(upper), int(arr.size))


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> ConfidenceInterval:
    """Wilson interval for bounded Bernoulli outcomes (e.g. release/failure rates)."""
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Require 0 <= successes <= total and total > 0")
    # 95% default; use standard-library inverse-normal approximation for common values.
    z_map = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    z = z_map.get(round(confidence, 2))
    if z is None:
        raise ValueError("Supported confidence values: 0.90, 0.95, 0.99")
    p = successes / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total)) / denom
    margin = z * math.sqrt((p*(1-p) + z*z/(4*total))/total) / denom
    return ConfidenceInterval(float(p), max(0.0, center-margin), min(1.0, center+margin), total)
