"""Chance-centered, fail-closed privacy gate for TAPF-MIN v2.2.

This module treats identity AUC symmetrically around chance. An AUC below 0.5 is not
automatically safer because reversing the attacker's score orientation yields 1-AUC.
The effective AUC is therefore 0.5 + abs(AUC - 0.5).

This is an empirical engineering gate, not a formal anonymity, differential-privacy,
or clinical-safety guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


def effective_auc(auc: float) -> float:
    """Convert AUC to attacker advantage above chance, independent of orientation."""
    a = float(auc)
    if not 0.0 <= a <= 1.0:
        raise ValueError("AUC must be in [0, 1]")
    return 0.5 + abs(a - 0.5)


def effective_auc_upper_from_ci(ci95: Sequence[float]) -> float:
    """Conservative effective AUC implied by a two-sided confidence interval."""
    if len(ci95) != 2:
        raise ValueError("ci95 must contain [low, high]")
    lo, hi = map(float, ci95)
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError("invalid AUC confidence interval")
    return 0.5 + max(abs(lo - 0.5), abs(hi - 0.5))


def identity_advantage(auc: float) -> float:
    """Absolute attacker advantage over random ranking."""
    return abs(float(auc) - 0.5)


@dataclass(frozen=True)
class V22PrivacyPolicy:
    """Predeclared empirical release policy for TAPF-MIN v2.2."""

    max_identity_advantage: float = 0.05
    min_task_f1_lower_ci: float = 0.20

    @property
    def max_effective_auc(self) -> float:
        return 0.5 + self.max_identity_advantage


@dataclass
class TemporalExposureAccumulator:
    """Fail-closed runtime guard for accumulated empirical identity evidence.

    The accumulator is deliberately labelled an engineering guard. It does not claim
    formal privacy composition. Publication evidence must still include an actual
    repeated-release attacker.
    """

    max_total_advantage: float = 0.10
    advantages: list[float] = field(default_factory=list)

    def add_auc(self, auc: float) -> float:
        adv = identity_advantage(auc)
        self.advantages.append(adv)
        return self.total_advantage

    @property
    def total_advantage(self) -> float:
        return float(sum(self.advantages))

    @property
    def allow_more(self) -> bool:
        return self.total_advantage <= self.max_total_advantage

    def reset(self) -> None:
        self.advantages.clear()


def evaluate_v22_gate(
    *,
    clip_auc_ci95: Sequence[float],
    repeated_release_auc: float,
    task_f1_ci95: Sequence[float],
    attacker_aucs: Iterable[float] | None = None,
    policy: V22PrivacyPolicy | None = None,
) -> dict:
    """Evaluate the complete v2.2 empirical release gate.

    RELEASE requires:
    - conservative clip-level effective AUC <= 0.55,
    - repeated-release effective AUC <= 0.55,
    - every supplied independent attacker <= 0.55 after chance-centering,
    - lower task Macro-F1 confidence bound >= 0.20.
    """
    p = policy or V22PrivacyPolicy()
    if len(task_f1_ci95) != 2:
        raise ValueError("task_f1_ci95 must contain [low, high]")

    clip_eff_upper = effective_auc_upper_from_ci(clip_auc_ci95)
    repeated_eff = effective_auc(repeated_release_auc)
    attacker_effective = [effective_auc(x) for x in (attacker_aucs or [])]
    attackers_pass = all(x <= p.max_effective_auc for x in attacker_effective)
    utility_lower = float(task_f1_ci95[0])

    checks = {
        "clip_privacy": clip_eff_upper <= p.max_effective_auc,
        "repeated_release_privacy": repeated_eff <= p.max_effective_auc,
        "all_independent_attackers": attackers_pass,
        "task_utility": utility_lower >= p.min_task_f1_lower_ci,
    }
    release = all(checks.values())
    return {
        "decision": "RELEASE" if release else "BLOCK",
        "release_eligible": release,
        "checks": checks,
        "policy": {
            "chance_identity_auc": 0.5,
            "max_identity_advantage": p.max_identity_advantage,
            "max_effective_auc": p.max_effective_auc,
            "min_task_f1_lower_ci": p.min_task_f1_lower_ci,
        },
        "observed": {
            "clip_effective_auc_upper": clip_eff_upper,
            "repeated_release_effective_auc": repeated_eff,
            "independent_attacker_effective_aucs": attacker_effective,
            "task_f1_lower_ci": utility_lower,
        },
    }
