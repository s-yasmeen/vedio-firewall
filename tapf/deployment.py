"""Deployment policy primitives for TAPF-MIN.

This compatibility module retains the v2 normalized-risk interface used by earlier
prototype tests and clients. The scientifically preferred chance-centered v2.2 gate
lives in ``tapf.deployment_v22``.

The task registry is the single source of truth for task-aware disclosure. Unknown
clinical tasks fail closed: TAPF-MIN must never fall back to protected video merely
because a task was not recognized.
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


TASK_REGISTRY = {
    "emotion": {
        "label": "Expression / emotion assessment",
        "purpose_example": "Assess affective facial expression without requiring raw identity",
        "representations": ("action-units", "expression-embedding", "protected-video"),
        "preferred_representation": "action-units",
        "side": "clinical-assessment",
    },
    "facial-expression": {
        "label": "Facial expression assessment",
        "purpose_example": "Measure task-relevant facial action patterns",
        "representations": ("action-units", "expression-embedding", "protected-video"),
        "preferred_representation": "action-units",
        "side": "clinical-assessment",
    },
    "movement": {
        "label": "Facial / motor movement assessment",
        "purpose_example": "Assess task-relevant movement while minimizing identity disclosure",
        "representations": ("landmark-trajectories", "motion-features", "protected-video"),
        "preferred_representation": "motion-features",
        "side": "clinical-assessment",
    },
    "neurology-motion": {
        "label": "Neurological movement assessment",
        "purpose_example": "Assess clinically relevant facial or motor dynamics",
        "representations": ("landmark-trajectories", "motion-features", "protected-video"),
        "preferred_representation": "motion-features",
        "side": "clinical-assessment",
    },
    "rppg": {
        "label": "Remote physiological signal (rPPG)",
        "purpose_example": "Estimate a permitted physiological signal without transmitting raw face video",
        "representations": ("physiological-signal", "protected-video"),
        "preferred_representation": "physiological-signal",
        "side": "clinical-assessment",
    },
    "authentication": {
        "label": "Patient authentication",
        "purpose_example": "Verify the enrolled patient using a protected cancelable biometric template",
        "representations": ("cancelable-template",),
        "preferred_representation": "cancelable-template",
        "side": "identity-verification",
    },
    "clinician-visual": {
        "label": "Clinician visual examination",
        "purpose_example": "Permit clinician viewing only when a visual video representation is necessary",
        "representations": ("protected-video",),
        "preferred_representation": "protected-video",
        "side": "human-visual-review",
    },
}

# Backward-compatible map for callers that only need representation tuples.
TASK_REPRESENTATIONS = {
    task: tuple(spec["representations"]) for task, spec in TASK_REGISTRY.items()
}


def task_registry() -> dict:
    """Return serializable task metadata for doctor-side request construction."""
    return {
        task: {
            **{k: v for k, v in spec.items() if k != "representations"},
            "representations": list(spec["representations"]),
        }
        for task, spec in TASK_REGISTRY.items()
    }


def is_known_task(task: str) -> bool:
    return task.strip().lower() in TASK_REGISTRY


def allowed_representations(task: str) -> tuple[str, ...]:
    """Return permitted representations; unknown tasks fail closed with no options."""
    spec = TASK_REGISTRY.get(task.strip().lower())
    return tuple(spec["representations"]) if spec else ()


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
    """Choose the least-disclosing valid operating point."""
    allowed = allowed_representations(task)
    if not allowed:
        return {
            "decision": "BLOCK",
            "reason": "unknown_or_unauthorized_task",
            "selected_alpha": None,
            "allowed_representations": (),
            "history": [],
        }
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
