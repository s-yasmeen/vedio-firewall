"""Evidence provenance manifest for TAPF-MIN."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
import json


@dataclass(frozen=True)
class EvidenceProvenance:
    code_commit: str
    model_version: str
    policy_version: str
    dataset_name: str
    dataset_version: str
    preprocessing_version: str
    attacker_version: str
    evaluator_id: str
    seed: int
    protocol_id: str

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def as_attestation_block(self) -> dict:
        return {**asdict(self), "provenance_sha256": self.digest()}
