"""Fail-closed pre-validation gate for TAPF-MIN v3.1/v3.4.

This module does not prove privacy or model quality. It prevents progression to final
validation unless independently measured evidence satisfies predeclared engineering and
scientific conditions. Missing evidence always BLOCKs.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PrevalidationPolicy:
    # Development utility must be materially better than the frozen v3 baseline.
    frozen_v3_macro_f1: float = 0.15050056895801578
    min_absolute_f1_gain: float = 0.20
    min_macro_f1_lcb: float = 0.55
    # Preprocessing must be reliable and not strongly actor-dependent.
    min_face_valid_rate: float = 0.90
    min_face_detection_rate: float = 0.80
    max_actor_detection_gap: float = 0.15
    # Empirical privacy gates remain separate from formal DP.
    max_identity_auc_ucb: float = 0.60
    max_repeated_auc_ucb: float = 0.60
    max_verification_auc_ucb: float = 0.60
    max_linkage_auc_ucb: float = 0.60
    # Formal release evidence.
    require_formal_dp: bool = True
    require_actor_disjoint: bool = True
    require_single_frozen_model: bool = True
    require_no_raw_or_latent_release: bool = True


REQUIRED_FIELDS = (
    'macro_f1', 'macro_f1_lcb', 'face_valid_rate', 'face_detection_rate',
    'actor_detection_gap', 'identity_auc_ucb', 'repeated_auc_ucb',
    'verification_auc_ucb', 'linkage_auc_ucb', 'actor_overlap',
    'single_frozen_model', 'formal_dp_applied', 'raw_video_released',
    'latent_released',
)


def evaluate_prevalidation(evidence: dict, policy: PrevalidationPolicy | None = None) -> dict:
    p = policy or PrevalidationPolicy()
    missing=[k for k in REQUIRED_FIELDS if k not in evidence]
    reasons=[]
    if missing:
        reasons.append('missing evidence: ' + ', '.join(sorted(missing)))
        return {'decision':'BLOCK','reasons':reasons,'missing':missing}

    def finite(name):
        try:
            return math.isfinite(float(evidence[name]))
        except Exception:
            return False

    numeric=[k for k in REQUIRED_FIELDS if k not in {
        'single_frozen_model','formal_dp_applied','raw_video_released','latent_released'
    }]
    bad=[k for k in numeric if not finite(k)]
    if bad:
        return {'decision':'BLOCK','reasons':['non-finite evidence: '+', '.join(sorted(bad))],'missing':[]}

    if float(evidence['macro_f1']) < p.frozen_v3_macro_f1 + p.min_absolute_f1_gain:
        reasons.append('FER improvement is not materially above frozen v3 baseline')
    if float(evidence['macro_f1_lcb']) < p.min_macro_f1_lcb:
        reasons.append('FER lower confidence bound below development gate')
    if float(evidence['face_valid_rate']) < p.min_face_valid_rate:
        reasons.append('valid face coverage below gate')
    if float(evidence['face_detection_rate']) < p.min_face_detection_rate:
        reasons.append('face detection coverage below gate')
    if float(evidence['actor_detection_gap']) > p.max_actor_detection_gap:
        reasons.append('actor-dependent face detection gap too large')
    for field,limit,label in (
        ('identity_auc_ucb',p.max_identity_auc_ucb,'clip identity'),
        ('repeated_auc_ucb',p.max_repeated_auc_ucb,'repeated release'),
        ('verification_auc_ucb',p.max_verification_auc_ucb,'verification'),
        ('linkage_auc_ucb',p.max_linkage_auc_ucb,'linkage'),
    ):
        if float(evidence[field]) > limit:
            reasons.append(f'{label} leakage exceeds empirical privacy gate')
    if p.require_actor_disjoint and int(evidence['actor_overlap']) != 0:
        reasons.append('actor overlap is non-zero')
    if p.require_single_frozen_model and not bool(evidence['single_frozen_model']):
        reasons.append('privacy audit did not use one frozen model')
    if p.require_formal_dp and not bool(evidence['formal_dp_applied']):
        reasons.append('formal release DP evidence missing')
    if p.require_no_raw_or_latent_release and bool(evidence['raw_video_released']):
        reasons.append('raw video release forbidden')
    if p.require_no_raw_or_latent_release and bool(evidence['latent_released']):
        reasons.append('task latent release forbidden')
    return {'decision':'PASS' if not reasons else 'BLOCK','reasons':reasons,'missing':[]}
