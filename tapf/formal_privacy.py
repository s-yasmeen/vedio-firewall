"""Formal privacy primitives for TAPF-MIN v3.

This module adds mathematically defined privacy mechanisms to the release layer.
It deliberately separates three concepts:

1. Differential privacy (formal): randomized output mechanisms with an explicit
   adjacency definition and privacy budget.
2. Minimum disclosure (data minimization): choose the smallest authorized output
   that meets task utility and policy constraints.
3. Empirical biometric privacy: independent identity/linkage attacks on the exact
   released object. Passing those attacks is evidence, not a theorem.

Formal scope
------------
``gaussian_release`` treats one bounded task representation as one record. The
input representation is L2-clipped to radius C, so replace-one adjacency has
L2 sensitivity at most 2C. With the classic Gaussian calibration used here,
each invocation is (epsilon, delta)-DP for 0 < epsilon <= 1.

``randomized_response`` releases one categorical task label and is exactly
``epsilon``-local-DP for any epsilon > 0.

``BasicDPAccountant`` uses the standard sequential-composition theorem:
composing mechanisms (eps_i, delta_i) yields
(sum eps_i, sum delta_i)-DP. This is conservative but explicit and auditable.

These guarantees do NOT imply biometric anonymity, clinical validity, protection
of the model-training set, or resistance to side channels outside the mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


class PrivacyBudgetExceeded(RuntimeError):
    """Raised when a requested formal-privacy release would exceed policy."""


@dataclass(frozen=True)
class PrivacySpend:
    mechanism: str
    epsilon: float
    delta: float


@dataclass(frozen=True)
class FormalPrivacyBudget:
    epsilon_max: float
    delta_max: float

    def __post_init__(self):
        if not math.isfinite(self.epsilon_max) or self.epsilon_max <= 0:
            raise ValueError("epsilon_max must be finite and > 0")
        if not math.isfinite(self.delta_max) or not 0 <= self.delta_max < 1:
            raise ValueError("delta_max must be finite and in [0,1)")


class BasicDPAccountant:
    """Fail-closed sequential-composition accountant.

    Basic composition is intentionally used as the default because it is simple,
    deterministic, and does not depend on an approximation library. A tighter RDP
    or PRV accountant can later be added as an optimization without weakening this
    baseline guarantee.
    """

    def __init__(self, budget: FormalPrivacyBudget):
        self.budget = budget
        self._spends: list[PrivacySpend] = []

    @property
    def spends(self) -> tuple[PrivacySpend, ...]:
        return tuple(self._spends)

    @property
    def epsilon_spent(self) -> float:
        return float(sum(s.epsilon for s in self._spends))

    @property
    def delta_spent(self) -> float:
        return float(sum(s.delta for s in self._spends))

    @property
    def remaining(self) -> dict:
        return {
            "epsilon": max(0.0, self.budget.epsilon_max - self.epsilon_spent),
            "delta": max(0.0, self.budget.delta_max - self.delta_spent),
        }

    def can_spend(self, epsilon: float, delta: float = 0.0) -> bool:
        _validate_spend(epsilon, delta)
        return (
            self.epsilon_spent + epsilon <= self.budget.epsilon_max + 1e-12
            and self.delta_spent + delta <= self.budget.delta_max + 1e-18
        )

    def spend(self, mechanism: str, epsilon: float, delta: float = 0.0) -> PrivacySpend:
        if not mechanism:
            raise ValueError("mechanism must be non-empty")
        if not self.can_spend(epsilon, delta):
            raise PrivacyBudgetExceeded(
                "Formal privacy budget exhausted: "
                f"requested (eps={epsilon}, delta={delta}); "
                f"spent (eps={self.epsilon_spent}, delta={self.delta_spent}); "
                f"budget (eps={self.budget.epsilon_max}, delta={self.budget.delta_max})"
            )
        item = PrivacySpend(str(mechanism), float(epsilon), float(delta))
        self._spends.append(item)
        return item

    def statement(self) -> dict:
        return {
            "accountant": "basic_sequential_composition",
            "epsilon_spent": self.epsilon_spent,
            "delta_spent": self.delta_spent,
            "epsilon_max": self.budget.epsilon_max,
            "delta_max": self.budget.delta_max,
            "release_count": len(self._spends),
            "guarantee": "Composed releases are (sum epsilon_i, sum delta_i)-DP under the stated per-mechanism adjacency assumptions.",
        }


def _validate_spend(epsilon: float, delta: float) -> None:
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError("epsilon must be finite and > 0")
    if not math.isfinite(float(delta)) or not 0 <= float(delta) < 1:
        raise ValueError("delta must be finite and in [0,1)")


def l2_clip(vector: Sequence[float], clip_norm: float = 1.0) -> np.ndarray:
    """L2 clip a vector to a public radius without changing its dimension."""
    if not math.isfinite(float(clip_norm)) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and > 0")
    x = np.asarray(vector, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("vector must be non-empty and finite")
    norm = float(np.linalg.norm(x))
    if norm > clip_norm:
        x = x * (float(clip_norm) / norm)
    return x


def gaussian_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    """Classic sufficient Gaussian-mechanism calibration.

    Uses sigma >= Delta_2 * sqrt(2 ln(1.25/delta)) / epsilon, valid for
    the classic theorem's 0 < epsilon <= 1 regime.
    """
    _validate_spend(epsilon, delta)
    if epsilon > 1:
        raise ValueError("classic Gaussian calibration requires epsilon <= 1")
    if delta <= 0:
        raise ValueError("Gaussian mechanism requires delta > 0")
    if not math.isfinite(float(sensitivity)) or sensitivity <= 0:
        raise ValueError("sensitivity must be finite and > 0")
    return float(sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon)


def gaussian_release(
    vector: Sequence[float],
    *,
    epsilon: float,
    delta: float,
    clip_norm: float = 1.0,
    rng: np.random.Generator | None = None,
    accountant: BasicDPAccountant | None = None,
) -> dict:
    """Release a clipped task vector with a formal (epsilon, delta)-DP guarantee.

    Adjacency: replacement of one arbitrary input record. Because both deterministic
    task vectors are clipped to radius C, Delta_2 <= 2C.
    """
    clipped = l2_clip(vector, clip_norm)
    sensitivity = 2.0 * float(clip_norm)
    sigma = gaussian_sigma(epsilon, delta, sensitivity)
    if accountant is not None:
        accountant.spend("gaussian_bounded_task_vector", epsilon, delta)
    generator = rng or np.random.default_rng()
    released = clipped + generator.normal(0.0, sigma, size=clipped.shape)
    return {
        "values": released.astype(np.float32),
        "mechanism": "gaussian",
        "epsilon": float(epsilon),
        "delta": float(delta),
        "clip_norm": float(clip_norm),
        "l2_sensitivity": sensitivity,
        "noise_sigma": sigma,
        "adjacency": "replace-one bounded task representation",
        "formal_guarantee": "(epsilon, delta)-DP for this invocation under the stated adjacency definition",
    }


def randomized_response(
    label: str,
    classes: Sequence[str],
    *,
    epsilon: float,
    rng: np.random.Generator | None = None,
    accountant: BasicDPAccountant | None = None,
) -> dict:
    """Release a categorical task label with exact epsilon-local-DP.

    For k classes, the true class is emitted with probability
    exp(epsilon)/(exp(epsilon)+k-1), and each other class with probability
    1/(exp(epsilon)+k-1).
    """
    _validate_spend(epsilon, 0.0)
    labels = tuple(str(x) for x in classes)
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("classes must contain at least two unique labels")
    if label not in labels:
        raise ValueError("label must occur in classes")
    exp_eps = math.exp(float(epsilon))
    denom = exp_eps + len(labels) - 1
    p_true = exp_eps / denom
    p_other = 1.0 / denom
    probs = np.full(len(labels), p_other, dtype=np.float64)
    probs[labels.index(label)] = p_true
    generator = rng or np.random.default_rng()
    released = str(generator.choice(np.asarray(labels), p=probs))
    if accountant is not None:
        accountant.spend("kary_randomized_response", epsilon, 0.0)
    return {
        "label": released,
        "mechanism": "kary_randomized_response",
        "epsilon": float(epsilon),
        "delta": 0.0,
        "n_classes": len(labels),
        "truth_probability": float(p_true),
        "other_probability": float(p_other),
        "adjacency": "any two possible private categorical inputs",
        "formal_guarantee": "epsilon-local-DP for this invocation",
    }


@dataclass(frozen=True)
class DisclosureCandidate:
    """One auditable output option in a minimum-disclosure lattice."""

    name: str
    disclosure_bits: int
    utility_lower: float
    empirical_identity_risk_upper: float
    formal_epsilon: float
    formal_delta: float = 0.0

    def __post_init__(self):
        if self.disclosure_bits <= 0:
            raise ValueError("disclosure_bits must be > 0")
        if not 0 <= self.utility_lower <= 1:
            raise ValueError("utility_lower must be in [0,1]")
        if not 0 <= self.empirical_identity_risk_upper <= 1:
            raise ValueError("empirical_identity_risk_upper must be in [0,1]")
        _validate_spend(self.formal_epsilon, self.formal_delta)


def select_minimum_disclosure(
    candidates: Iterable[DisclosureCandidate],
    *,
    min_utility_lower: float,
    max_empirical_identity_risk_upper: float,
    accountant: BasicDPAccountant,
) -> dict:
    """Choose the smallest measured output that satisfies all release constraints.

    ``disclosure_bits`` is an explicit engineering data-minimization metric, not a
    privacy theorem. The DP guarantee is supplied separately by the candidate's
    randomized mechanism and the accountant.
    """
    rows = sorted(candidates, key=lambda c: (c.disclosure_bits, c.formal_epsilon, c.name))
    history = []
    for c in rows:
        utility_ok = c.utility_lower >= min_utility_lower
        empirical_ok = c.empirical_identity_risk_upper <= max_empirical_identity_risk_upper
        budget_ok = accountant.can_spend(c.formal_epsilon, c.formal_delta)
        row = {
            "name": c.name,
            "disclosure_bits": c.disclosure_bits,
            "utility_lower": c.utility_lower,
            "empirical_identity_risk_upper": c.empirical_identity_risk_upper,
            "formal_epsilon": c.formal_epsilon,
            "formal_delta": c.formal_delta,
            "utility_pass": utility_ok,
            "empirical_privacy_pass": empirical_ok,
            "formal_budget_pass": budget_ok,
        }
        row["eligible"] = bool(utility_ok and empirical_ok and budget_ok)
        history.append(row)
        if row["eligible"]:
            return {
                "decision": "RELEASE",
                "selected": c.name,
                "disclosure_bits": c.disclosure_bits,
                "history": history,
            }
    return {"decision": "BLOCK", "selected": None, "history": history}
