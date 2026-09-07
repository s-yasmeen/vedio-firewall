"""TAPF-MIN adaptive minimum-disclosure release controller."""
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, Optional

@dataclass(frozen=True)
class ReleasePolicy:
    privacy_threshold: float = 0.25
    utility_threshold: float = 0.75
    temporal_threshold: float = 0.30
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
    Risk and utility are supplied by independent evaluator adapters.
    """

    def __init__(self, policy: ReleasePolicy):
        self.policy = policy

    def evaluate(self, alpha: float, identity_risk: float,
                 task_utility: float, temporal_risk: float) -> Evaluation:
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
            identity_risk, task_utility, temporal_risk = evaluator(alpha)
            result = self.evaluate(alpha, identity_risk, task_utility, temporal_risk)
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
