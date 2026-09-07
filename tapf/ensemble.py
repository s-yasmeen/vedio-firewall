"""Worst-case privacy evidence aggregation across independent attackers."""
from __future__ import annotations
from dataclasses import replace
from typing import Mapping, Sequence
from .deployment import BoundedEvidence


def aggregate_worst_case(evidence_by_attacker: Mapping[str, Sequence[BoundedEvidence]]) -> list[BoundedEvidence]:
    if not evidence_by_attacker:
        raise ValueError("At least one attacker is required")
    alpha_maps = {name: {round(e.alpha, 8): e for e in rows} for name, rows in evidence_by_attacker.items()}
    common = set.intersection(*(set(m) for m in alpha_maps.values()))
    if not common:
        raise ValueError("Attackers do not share any alpha operating points")
    out=[]
    for alpha in sorted(common):
        rows=[m[alpha] for m in alpha_maps.values()]
        # Privacy/temporal use worst attacker; utility uses most conservative lower bound;
        # latency uses worst p95-equivalent evidence.
        template=rows[0]
        out.append(replace(
            template,
            identity_risk=max(r.identity_risk for r in rows),
            identity_risk_upper=max(r.identity_risk_upper for r in rows),
            task_utility=min(r.task_utility for r in rows),
            task_utility_lower=min(r.task_utility_lower for r in rows),
            temporal_risk=max(r.temporal_risk for r in rows),
            temporal_risk_upper=max(r.temporal_risk_upper for r in rows),
            latency_ms=max(r.latency_ms for r in rows),
            evaluator_id="worst-case:" + "+".join(sorted(evidence_by_attacker)),
            sample_count=min(r.sample_count for r in rows),
        ))
    return out
