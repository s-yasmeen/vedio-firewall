"""Confidence-aware resilience for multimodal TAPF-MIN deployments.

This module manages source health before privacy/utility release evaluation. It never
converts an unsafe privacy decision into a release. Its role is to decide whether the
available input set is healthy enough to produce evidence at all, and whether operation
is NORMAL, DEGRADED, or BLOCK.

It is an engineering resilience layer, not a clinical-safety certification.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence
import math
import numpy as np


@dataclass(frozen=True)
class SourceHealth:
    name: str
    quality: float
    available: bool = True
    base_weight: float = 1.0
    required: bool = False

    def validated_quality(self) -> float:
        q = float(self.quality)
        if not math.isfinite(q) or q < 0.0 or q > 1.0:
            return 0.0
        return q if self.available else 0.0


@dataclass(frozen=True)
class ResiliencePolicy:
    min_source_quality: float = 0.35
    normal_confidence: float = 0.80
    degraded_confidence: float = 0.55
    min_usable_sources: int = 1


@dataclass(frozen=True)
class SourceVector:
    health: SourceHealth
    values: Sequence[float]


def assess_resilience(sources: Iterable[SourceHealth], policy: ResiliencePolicy | None = None) -> dict:
    p = policy or ResiliencePolicy()
    rows = list(sources)
    if not rows:
        return {"mode":"BLOCK","confidence":0.0,"usable_sources":[],"failed_sources":[],"reason":"no_sources"}

    health_rows = []
    usable = []
    failed = []
    required_failed = []
    weighted_quality = 0.0
    total_weight = 0.0

    for src in rows:
        q = src.validated_quality()
        w = max(0.0, float(src.base_weight))
        ok = src.available and q >= p.min_source_quality and w > 0.0
        health_rows.append({**asdict(src), "effective_quality":q, "usable":ok})
        if ok:
            usable.append(src.name)
            weighted_quality += q * w
            total_weight += w
        else:
            failed.append(src.name)
            if src.required:
                required_failed.append(src.name)

    confidence = weighted_quality / total_weight if total_weight > 0 else 0.0
    if required_failed:
        mode, reason = "BLOCK", "required_source_unavailable"
    elif len(usable) < p.min_usable_sources:
        mode, reason = "BLOCK", "insufficient_usable_sources"
    elif confidence >= p.normal_confidence and not failed:
        mode, reason = "NORMAL", "all_sources_healthy"
    elif confidence >= p.degraded_confidence:
        mode, reason = "DEGRADED", "fallback_sources_active"
    else:
        mode, reason = "BLOCK", "confidence_below_degraded_threshold"

    return {
        "mode":mode,
        "confidence":float(confidence),
        "usable_sources":usable,
        "failed_sources":failed,
        "required_failed_sources":required_failed,
        "reason":reason,
        "policy":asdict(p),
        "sources":health_rows,
    }


def reliability_weighted_fusion(sources: Iterable[SourceVector], policy: ResiliencePolicy | None = None) -> dict:
    """Fuse equal-dimensional source vectors using health-adjusted weights.

    Unavailable or low-quality sources contribute zero weight. The function returns a
    BLOCK result rather than fabricating a vector if the resilience policy is not met.
    """
    rows = list(sources)
    assessment = assess_resilience([x.health for x in rows], policy)
    if assessment["mode"] == "BLOCK":
        return {**assessment, "fused":None, "effective_weights":{}}

    vectors = []
    weights = []
    names = []
    dim = None
    min_q = (policy or ResiliencePolicy()).min_source_quality
    for item in rows:
        q = item.health.validated_quality()
        if not item.health.available or q < min_q or item.health.base_weight <= 0:
            continue
        v = np.asarray(item.values, dtype=np.float64)
        if v.ndim != 1 or not np.isfinite(v).all():
            continue
        if dim is None:
            dim = v.size
        if v.size != dim:
            raise ValueError("all usable source vectors must have the same dimension")
        w = q * float(item.health.base_weight)
        vectors.append(v); weights.append(w); names.append(item.health.name)

    if not vectors or sum(weights) <= 0:
        return {**assessment, "mode":"BLOCK", "reason":"no_valid_vectors", "fused":None, "effective_weights":{}}

    W = np.asarray(weights, dtype=np.float64)
    W /= W.sum()
    fused = np.sum(np.stack(vectors, axis=0) * W[:,None], axis=0)
    return {
        **assessment,
        "fused":fused.astype(float).tolist(),
        "effective_weights":{name:float(w) for name,w in zip(names,W)},
    }


def gate_with_resilience(resilience: dict, privacy_utility_decision: dict) -> dict:
    """Combine source-health status with an existing TAPF privacy/utility decision.

    Resilience can only preserve or reduce permission: it can never turn BLOCK into
    RELEASE. DEGRADED can release only when the independent privacy/utility gate already
    returned RELEASE.
    """
    pu = str(privacy_utility_decision.get("decision", "BLOCK")).upper()
    mode = str(resilience.get("mode", "BLOCK")).upper()
    release = pu == "RELEASE" and mode in {"NORMAL", "DEGRADED"}
    reason = "all_gates_pass" if release else (
        "resilience_block" if mode == "BLOCK" else "privacy_or_utility_block"
    )
    return {
        "decision":"RELEASE" if release else "BLOCK",
        "operating_mode":mode,
        "system_confidence":float(resilience.get("confidence", 0.0)),
        "reason":reason,
        "resilience":resilience,
        "privacy_utility":privacy_utility_decision,
    }
