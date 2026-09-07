"""Deployment policy primitives for TAPF-MIN.

This module keeps deployment release decisions separate from experimental image
transforms. Production decisions use conservative confidence bounds, latency,
and a task-conditioned representation policy.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class DeploymentPolicy:
    privacy_upper_threshold: float = 0.25
    utility_lower_threshold: float = 0.75
    temporal_upper_threshold: float = 0.30
    latency_ms_threshold: float = 150.0
    fail_closed: bool = True


@dataclass(frozen=True)
class BoundedEvidence:
    alpha: float
    identity_risk: float
    identity_risk_upper: float
    task_utility: float
    task_utility_lower: float
    temporal_risk: float
    temporal_risk_upper: float
    latency_ms: float
    evaluator_id: str = "unknown"
    sample_count: int = 0


TASK_REPRESENTATIONS = {
    "facial-expression": ("action-units", "expression-embedding", "protected-video"),
    "emotion": ("action-units", "expression-embedding", "protected-video"),
    "movement": ("landmark-trajectories", "motion-features", "protected-video"),
    "neurology-motion": ("landmark-trajectories", "motion-features", "protected-video"),
    "rppg": ("physiological-signal", "protected-video"),
    "authentication": ("cancelable-template",),
    "clinician-visual": ("protected-video",),
}


def allowed_representations(task: str) -> tuple[str, ...]:
    return TASK_REPRESENTATIONS.get(task.strip().lower(), ("protected-video",))


def representation_rank(task: str, representation: str) -> int:
    reps = allowed_representations(task)
    try:
        return reps.index(representation)
    except ValueError:
        return 10_000


def evaluate_bounded(e: BoundedEvidence, policy: DeploymentPolicy) -> dict:
    checks = {
        "privacy_pass": e.identity_risk_upper <= policy.privacy_upper_threshold,
        "utility_pass": e.task_utility_lower >= policy.utility_lower_threshold,
        "temporal_pass": e.temporal_risk_upper <= policy.temporal_upper_threshold,
        "latency_pass": e.latency_ms <= policy.latency_ms_threshold,
        "evidence_complete": e.sample_count > 0 and bool(e.evaluator_id),
    }
    return {**asdict(e), **checks, "release": all(checks.values())}


def select_minimum_release(task: str, representation: str,
                           evidence: Iterable[BoundedEvidence],
                           policy: DeploymentPolicy) -> dict:
    """Choose the least-disclosing valid operating point.

    Representation is validated against the task policy. Within a representation,
    the minimum alpha satisfying all conservative criteria is selected.
    """
    allowed = allowed_representations(task)
    if representation not in allowed:
        return {
            "decision": "BLOCK",
            "reason": "representation_not_authorized_for_task",
            "selected_alpha": None,
            "allowed_representations": allowed,
            "history": [],
        }

    rows = sorted(evidence, key=lambda x: x.alpha)
    history = []
    for row in rows:
        result = evaluate_bounded(row, policy)
        history.append(result)
        if result["release"]:
            return {
                "decision": "RELEASE",
                "reason": "bounded_constraints_satisfied",
                "selected_alpha": row.alpha,
                "evaluation": result,
                "history": history,
                "allowed_representations": allowed,
            }
    return {
        "decision": "BLOCK" if policy.fail_closed else "UNVERIFIED",
        "reason": "no_bounded_operating_point_satisfied_policy",
        "selected_alpha": None,
        "evaluation": None,
        "history": history,
        "allowed_representations": allowed,
    }
