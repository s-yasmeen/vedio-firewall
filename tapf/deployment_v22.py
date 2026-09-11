"""Deployment-grade adapter for the TAPF-MIN v2.2 empirical release gate.

This module keeps the competition/research privacy definition consistent with runtime
release decisions. It does not provide a formal anonymity, differential-privacy, or
clinical-safety guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

from tapf.deployment import allowed_representations
from tapf.privacy_gate_v22 import V22PrivacyPolicy, evaluate_v22_gate


@dataclass(frozen=True)
class V22DeploymentPolicy:
    max_identity_advantage: float = 0.05
    min_task_f1_lower_ci: float = 0.20
    latency_ms_threshold: float = 150.0
    fail_closed: bool = True

    @property
    def max_effective_auc(self) -> float:
        return 0.5 + self.max_identity_advantage


@dataclass(frozen=True)
class V22BoundedEvidence:
    alpha: float
    clip_auc_ci95_low: float
    clip_auc_ci95_high: float
    repeated_release_auc: float
    task_f1_ci95_low: float
    task_f1_ci95_high: float
    attacker_aucs: tuple[float, ...]
    latency_ms: float
    evaluator_id: str
    sample_count: int


def evaluate_v22_evidence(e: V22BoundedEvidence, policy: V22DeploymentPolicy) -> dict:
    gate = evaluate_v22_gate(
        clip_auc_ci95=[e.clip_auc_ci95_low, e.clip_auc_ci95_high],
        repeated_release_auc=e.repeated_release_auc,
        task_f1_ci95=[e.task_f1_ci95_low, e.task_f1_ci95_high],
        attacker_aucs=e.attacker_aucs,
        policy=V22PrivacyPolicy(
            max_identity_advantage=policy.max_identity_advantage,
            min_task_f1_lower_ci=policy.min_task_f1_lower_ci,
        ),
    )
    latency_pass = float(e.latency_ms) <= policy.latency_ms_threshold
    evidence_complete = e.sample_count > 0 and bool(e.evaluator_id) and len(e.attacker_aucs) > 0
    release = bool(gate["release_eligible"] and latency_pass and evidence_complete)
    return {
        **asdict(e),
        "privacy_utility_gate": gate,
        "latency_pass": latency_pass,
        "evidence_complete": evidence_complete,
        "release": release,
    }


def select_v22_release(task: str, representation: str,
                       evidence: Sequence[V22BoundedEvidence],
                       policy: V22DeploymentPolicy | None = None) -> dict:
    p = policy or V22DeploymentPolicy()
    allowed = allowed_representations(task)
    if representation not in allowed:
        return {
            "decision": "BLOCK",
            "reason": "representation_not_authorized_for_task",
            "selected_alpha": None,
            "allowed_representations": allowed,
            "history": [],
        }

    history = []
    for row in sorted(evidence, key=lambda x: x.alpha):
        result = evaluate_v22_evidence(row, p)
        history.append(result)
        if result["release"]:
            return {
                "decision": "RELEASE",
                "reason": "v22_chance_centered_constraints_satisfied",
                "selected_alpha": row.alpha,
                "evaluation": result,
                "history": history,
                "allowed_representations": allowed,
            }

    return {
        "decision": "BLOCK" if p.fail_closed else "UNVERIFIED",
        "reason": "no_v22_operating_point_satisfied_policy",
        "selected_alpha": None,
        "evaluation": None,
        "history": history,
        "allowed_representations": allowed,
    }
