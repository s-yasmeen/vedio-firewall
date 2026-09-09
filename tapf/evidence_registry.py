"""Frozen, versioned release evidence for TAPF-MIN v3.

Release authorization must use server-controlled evidence bound to a model and protocol,
never utility/privacy numbers supplied by the release caller.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, json
from pathlib import Path

@dataclass(frozen=True)
class FrozenEvidence:
    evidence_id: str
    model_sha256: str
    protocol_sha256: str
    task: str
    mechanism: str
    utility_lower: float
    identity_risk_upper: float
    repeated_risk_upper: float
    epsilon: float
    delta: float = 0.0
    quantization_bits: int | None = None
    frozen: bool = True

    def __post_init__(self):
        if not self.evidence_id or not self.task: raise ValueError('evidence_id/task required')
        for h in (self.model_sha256, self.protocol_sha256):
            if len(h) != 64 or any(c not in '0123456789abcdef' for c in h.lower()):
                raise ValueError('hashes must be 64-char SHA-256 hex')
        for x in (self.utility_lower, self.identity_risk_upper, self.repeated_risk_upper):
            if not 0 <= x <= 1: raise ValueError('bounded metrics must be in [0,1]')
        if self.epsilon <= 0 or not 0 <= self.delta < 1: raise ValueError('invalid DP parameters')
        if not self.frozen: raise ValueError('release evidence must be frozen')

    @property
    def digest(self) -> str:
        raw=json.dumps(asdict(self),sort_keys=True,separators=(',',':')).encode()
        return hashlib.sha256(raw).hexdigest()

class EvidenceRegistry:
    def __init__(self, entries):
        self._entries={e.evidence_id:e for e in entries}
        if len(self._entries)!=len(list(entries)):
            raise ValueError('duplicate evidence_id')

    @classmethod
    def from_json(cls, path: str | Path):
        payload=json.loads(Path(path).read_text())
        if payload.get('schema')!='tapf-min-evidence/v1': raise ValueError('unsupported evidence schema')
        return cls([FrozenEvidence(**row) for row in payload.get('entries',[])])

    def get(self, evidence_id: str, *, task: str | None=None) -> FrozenEvidence:
        try: e=self._entries[evidence_id]
        except KeyError: raise KeyError('unknown frozen evidence_id')
        if task is not None and e.task != task: raise ValueError('evidence task mismatch')
        return e
