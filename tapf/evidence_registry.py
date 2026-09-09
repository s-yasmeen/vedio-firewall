"""Versioned, fail-closed evidence registry for TAPF-MIN v3.

Release decisions must rely on frozen server-side evidence, never on utility/privacy
numbers supplied by the requesting client. Each entry binds a model hash, protocol
hash, representation, mechanism, confidence-bound evidence, and formal privacy spend.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FrozenEvidence:
    evidence_id: str
    task: str
    representation: str
    mechanism: str
    model_sha256: str
    protocol_sha256: str
    utility_lower: float
    identity_risk_upper: float
    repeated_release_risk_upper: float
    epsilon: float
    delta: float = 0.0
    disclosure_bits: int = 1
    status: str = "approved"

    def __post_init__(self):
        if not self.evidence_id or not self.task or not self.representation:
            raise ValueError("evidence_id, task, and representation are required")
        for name, value in {
            "utility_lower": self.utility_lower,
            "identity_risk_upper": self.identity_risk_upper,
            "repeated_release_risk_upper": self.repeated_release_risk_upper,
        }.items():
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.epsilon <= 0 or not 0 <= self.delta < 1:
            raise ValueError("invalid formal privacy parameters")
        if self.disclosure_bits <= 0:
            raise ValueError("disclosure_bits must be >0")
        for name, value in {"model_sha256": self.model_sha256, "protocol_sha256": self.protocol_sha256}.items():
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
                raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
        if self.status not in {"approved", "blocked", "retired"}:
            raise ValueError("status must be approved, blocked, or retired")

    def canonical_digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


class EvidenceRegistry:
    def __init__(self, entries: Iterable[FrozenEvidence]):
        items = list(entries)
        self._by_id = {e.evidence_id: e for e in items}
        if len(self._by_id) != len(items):
            raise ValueError("evidence IDs must be unique")

    @classmethod
    def from_json(cls, path: str | Path) -> "EvidenceRegistry":
        raw = json.loads(Path(path).read_text())
        if raw.get("schema") != "tapf-min-evidence-registry/v1":
            raise ValueError("unsupported evidence registry schema")
        return cls(FrozenEvidence(**row) for row in raw.get("entries", []))

    def get(self, evidence_id: str) -> FrozenEvidence:
        try:
            item = self._by_id[str(evidence_id)]
        except KeyError as exc:
            raise KeyError(f"unknown evidence_id: {evidence_id}") from exc
        if item.status != "approved":
            raise PermissionError(f"evidence is not approved: {item.status}")
        return item

    def eligible(
        self,
        *,
        task: str,
        model_sha256: str,
        protocol_sha256: str,
        min_utility_lower: float,
        max_identity_risk_upper: float,
        max_repeated_release_risk_upper: float,
    ) -> list[FrozenEvidence]:
        rows = []
        for e in self._by_id.values():
            if e.status != "approved" or e.task != task:
                continue
            if e.model_sha256 != model_sha256 or e.protocol_sha256 != protocol_sha256:
                continue
            if e.utility_lower < min_utility_lower:
                continue
            if e.identity_risk_upper > max_identity_risk_upper:
                continue
            if e.repeated_release_risk_upper > max_repeated_release_risk_upper:
                continue
            rows.append(e)
        return sorted(rows, key=lambda e: (e.disclosure_bits, e.epsilon, e.evidence_id))

    def snapshot_digest(self) -> str:
        rows = [asdict(self._by_id[k]) for k in sorted(self._by_id)]
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()
