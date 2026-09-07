"""Identity-risk evaluator interfaces for TAPF-MIN."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np


class IdentityAttacker(ABC):
    name: str = "identity-attacker"

    @abstractmethod
    def risk(self, frames: Sequence[np.ndarray], subject_id: str | None = None) -> float:
        """Return empirical identity risk in [0,1]; larger means more identity leakage."""
        raise NotImplementedError


class CosineEmbeddingAttacker(IdentityAttacker):
    """Generic verification attacker around an embedding function.

    This adapter supports real face-recognition backends while keeping the TAPF controller
    independent of a specific model. Enroll reference embeddings with ``enroll`` and then
    score transformed frames against the claimed identity.
    """

    def __init__(self, embedder, name="cosine-embedding-attacker"):
        self.embedder = embedder
        self.name = name
        self.references = {}

    @staticmethod
    def _normalize(v):
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(v) + 1e-12
        return v / n

    def enroll(self, subject_id: str, frames: Sequence[np.ndarray]):
        embs = [self._normalize(self.embedder(f)) for f in frames]
        if not embs:
            raise ValueError("No enrollment frames")
        self.references[subject_id] = self._normalize(np.mean(embs, axis=0))

    def risk(self, frames: Sequence[np.ndarray], subject_id: str | None = None) -> float:
        if subject_id is None or subject_id not in self.references:
            raise ValueError("subject_id must be enrolled before scoring")
        ref = self.references[subject_id]
        scores = []
        for frame in frames:
            emb = self._normalize(self.embedder(frame))
            cosine = float(np.dot(ref, emb))
            scores.append((cosine + 1.0) / 2.0)
        return float(max(scores)) if scores else 1.0
