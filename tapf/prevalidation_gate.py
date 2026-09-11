"""Fail-closed pre-validation gate for TAPF-MIN v3.1/v3.4.

This gate does not prove privacy or clinical validity. It prevents progression to final
validation unless independently measured evidence satisfies predeclared utility,
preprocessing, privacy, protocol-separation and release-safety conditions. Missing or
non-finite evidence always BLOCKs.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PrevalidationPolicy:
    frozen_v3_macro_f1: float = 0.15050056895801578
    min_absolute_f1_gain: float = 0.20
    min_macro_f1_lcb: float = 0.55
    min_face_valid_rate: float = 0.90
    min_face_detection_rate: float = 0.80
    min_eye_alignment_rate: float = 0.45
    max_bbox_reuse_rate: float = 0.20
    max_actor_detection_gap: float = 0.15
    min_audit_actors: int = 12
    max_identity_auc_ucb: float = 0.60
    max_repeated_auc_ucb: float = 0.60
    max_verification_auc_ucb: float = 0.60
    max_linkage_auc_ucb: float = 0.60
    require_formal_dp: bool = True
    require_actor_disjoint: bool = True
    require_single_frozen_model: bool = True
    require_no_raw_or_latent_release: bool = True
    require_selection_separate_from_final: bool = True
    require_actor_clustered_utility_ci: bool = True
    require_identity_level_privacy_ci: bool = True


REQUIRED_FIELDS = (
    "macro_f1",
    "macro_f1_lcb",
    "face_valid_rate",
    "face_detection_rate",
    "eye_alignment_rate",
    "bbox_reuse_rate",
    "actor_detection_gap",
    "audit_actor_count",
    "identity_auc_ucb",
    "repeated_auc_ucb",
    "verification_auc_ucb",
    "linkage_auc_ucb",
    "actor_overlap",
    "single_frozen_model",
    "selection_separate_from_final",
    "actor_clustered_utility_ci",
    "identity_level_privacy_ci",
    "formal_dp_applied",
    "raw_video_released",
    "latent_released",
)

_BOOLEAN_FIELDS = {
    "single_frozen_model",
    "selection_separate_from_final",
    "actor_clustered_utility_ci",
    "identity_level_privacy_ci",
    "formal_dp_applied",
    "raw_video_released",
    "latent_released",
}


def evaluate_prevalidation(evidence: dict, policy: PrevalidationPolicy | None = None) -> dict:
    p = policy or PrevalidationPolicy()
    missing = [k for k in REQUIRED_FIELDS if k not in evidence]
    if missing:
        return {
            "decision": "BLOCK",
            "reasons": ["missing evidence: " + ", ".join(sorted(missing))],
            "missing": missing,
        }

    numeric = [k for k in REQUIRED_FIELDS if k not in _BOOLEAN_FIELDS]
    bad = []
    for name in numeric:
        try:
            if not math.isfinite(float(evidence[name])):
                bad.append(name)
        except Exception:
            bad.append(name)
    if bad:
        return {
            "decision": "BLOCK",
            "reasons": ["non-finite evidence: " + ", ".join(sorted(bad))],
            "missing": [],
        }

    reasons = []
    if float(evidence["macro_f1"]) < p.frozen_v3_macro_f1 + p.min_absolute_f1_gain:
        reasons.append("FER improvement is not materially above frozen v3 baseline")
    if float(evidence["macro_f1_lcb"]) < p.min_macro_f1_lcb:
        reasons.append("FER lower confidence bound below development gate")
    if float(evidence["face_valid_rate"]) < p.min_face_valid_rate:
        reasons.append("valid face coverage below gate")
    if float(evidence["face_detection_rate"]) < p.min_face_detection_rate:
        reasons.append("face detection coverage below gate")
    if float(evidence["eye_alignment_rate"]) < p.min_eye_alignment_rate:
        reasons.append("eye alignment coverage below gate")
    if float(evidence["bbox_reuse_rate"]) > p.max_bbox_reuse_rate:
        reasons.append("bounding-box reuse rate exceeds gate")
    if float(evidence["actor_detection_gap"]) > p.max_actor_detection_gap:
        reasons.append("actor-dependent face detection gap too large")
    if int(evidence["audit_actor_count"]) < p.min_audit_actors:
        reasons.append("privacy audit actor cohort too small")

    for field, limit, label in (
        ("identity_auc_ucb", p.max_identity_auc_ucb, "clip identity"),
        ("repeated_auc_ucb", p.max_repeated_auc_ucb, "repeated release"),
        ("verification_auc_ucb", p.max_verification_auc_ucb, "verification"),
        ("linkage_auc_ucb", p.max_linkage_auc_ucb, "linkage"),
    ):
        if float(evidence[field]) > limit:
            reasons.append(f"{label} leakage exceeds empirical privacy gate")

    if p.require_actor_disjoint and int(evidence["actor_overlap"]) != 0:
        reasons.append("actor overlap is non-zero")
    if p.require_single_frozen_model and not bool(evidence["single_frozen_model"]):
        reasons.append("privacy audit did not use one frozen model")
    if p.require_selection_separate_from_final and not bool(evidence["selection_separate_from_final"]):
        reasons.append("architecture selection and final audit are not separated")
    if p.require_actor_clustered_utility_ci and not bool(evidence["actor_clustered_utility_ci"]):
        reasons.append("utility confidence interval is not actor-clustered")
    if p.require_identity_level_privacy_ci and not bool(evidence["identity_level_privacy_ci"]):
        reasons.append("privacy confidence interval is not identity-level/hierarchical")
    if p.require_formal_dp and not bool(evidence["formal_dp_applied"]):
        reasons.append("formal release DP evidence missing")
    if p.require_no_raw_or_latent_release and bool(evidence["raw_video_released"]):
        reasons.append("raw video release forbidden")
    if p.require_no_raw_or_latent_release and bool(evidence["latent_released"]):
        reasons.append("task latent release forbidden")

    return {"decision": "PASS" if not reasons else "BLOCK", "reasons": reasons, "missing": []}
