"""TAPF-MIN adaptive minimum-disclosure release controller.

The generic controller uses normalized empirical risk values. For TAPF-MIN v2.2 these
are interpreted as identity advantage over chance and repeated-release advantage, while
task utility is Macro-F1. The dedicated chance-centered AUC implementation lives in
``tapf.privacy_gate_v22`` / ``tapf.deployment_v22`` and should be preferred for final
release evidence.
"""
from dataclasses import dataclass, asdict
from typing import Callable


@dataclass(frozen=True)
class ReleasePolicy:
    # v2.2-compatible defaults: identity/repeated-release advantage <= 0.05,
    # task Macro-F1 lower bound >= 0.20. Explicit task-specific calibration may override.
    privacy_threshold: float = 0.05
    utility_threshold: float = 0.20
    temporal_threshold: float = 0.05
    alphas: tuple = tuple(i / 20 for i in range(21))
    fail_closed: bool = True


@dataclass
class Evaluation:
    alpha: float
    identity_risk: float
    task_utility: float
    temporal_risk: float
    privacy_pass: bool
    utility_pass: bool
    temporal_pass: bool

    @property
    def release(self):
        return self.privacy_pass and self.utility_pass and self.temporal_pass


class MinimumDisclosureController:
    """Select the minimum alpha satisfying all empirical release constraints.

    The controller deliberately does not assume that a transformed face is private.
    Risk and utility are supplied by independent evaluator adapters. It fails closed when
    no operating point meets all configured constraints.
    """

    def __init__(self, policy: ReleasePolicy):
        self.policy = policy

    def evaluate(self, alpha: float, identity_risk: float,
                 task_utility: float, temporal_risk: float) -> Evaluation:
        vals = (alpha, identity_risk, task_utility, temporal_risk)
        if any(not (float(v) == float(v)) for v in vals):
            raise ValueError("NaN evidence is not permitted")
        if not (0.0 <= float(identity_risk) <= 1.0 and
                0.0 <= float(task_utility) <= 1.0 and
                0.0 <= float(temporal_risk) <= 1.0):
            raise ValueError("risk/utility evidence must be in [0,1]")
        return Evaluation(
            alpha=float(alpha),
            identity_risk=float(identity_risk),
            task_utility=float(task_utility),
            temporal_risk=float(temporal_risk),
            privacy_pass=identity_risk <= self.policy.privacy_threshold,
            utility_pass=task_utility >= self.policy.utility_threshold,
            temporal_pass=temporal_risk <= self.policy.temporal_threshold,
        )

    def search(self, evaluator: Callable[[float], tuple]):
        history = []
        for alpha in self.policy.alphas:
            try:
                identity_risk, task_utility, temporal_risk = evaluator(alpha)
                result = self.evaluate(alpha, identity_risk, task_utility, temporal_risk)
            except Exception as exc:
                history.append({
                    "alpha": float(alpha), "release": False,
                    "error": f"invalid_or_missing_evidence:{type(exc).__name__}",
                })
                continue
            history.append(asdict(result) | {"release": result.release})
            if result.release:
                return {
                    "decision": "RELEASE",
                    "selected_alpha": alpha,
                    "evaluation": asdict(result),
                    "history": history,
                }
        return {
            "decision": "BLOCK" if self.policy.fail_closed else "UNVERIFIED",
            "selected_alpha": None,
            "evaluation": None,
            "history": history,
        }
