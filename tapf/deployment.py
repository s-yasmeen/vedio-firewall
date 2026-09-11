"""Deployment policy primitives for TAPF-MIN.

This module keeps deployment release decisions separate from experimental image
transforms. Generic normalized-risk decisions use v2.2-compatible defaults:
identity advantage <= 0.05, task Macro-F1 lower bound >= 0.20, and repeated-release
advantage <= 0.05. Direct chance-centered AUC evidence should use tapf.deployment_v22.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class DeploymentPolicy:
    privacy_upper_threshold: float = 0.05
    utility_lower_threshold: float = 0.20
    temporal_upper_threshold: float = 0.05
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


def _finite01(value: float) -> bool:
    v = float(value)
    return v == v and 0.0 <= v <= 1.0


def evaluate_bounded(e: BoundedEvidence, policy: DeploymentPolicy) -> dict:
    valid_numeric = all(_finite01(v) for v in (
        e.identity_risk, e.identity_risk_upper, e.task_utility, e.task_utility_lower,
        e.temporal_risk, e.temporal_risk_upper,
    )) and float(e.latency_ms) == float(e.latency_ms) and e.latency_ms >= 0.0
    checks = {
        "numeric_evidence_valid": valid_numeric,
        "privacy_pass": valid_numeric and e.identity_risk_upper <= policy.privacy_upper_threshold,
        "utility_pass": valid_numeric and e.task_utility_lower >= policy.utility_lower_threshold,
        "temporal_pass": valid_numeric and e.temporal_risk_upper <= policy.temporal_upper_threshold,
        "latency_pass": valid_numeric and e.latency_ms <= policy.latency_ms_threshold,
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
