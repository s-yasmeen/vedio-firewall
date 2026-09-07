"""Cancelable biometric-template research primitives.

This module is for empirical template-protection evaluation, not a cryptographic proof.
It provides keyed orthogonal projections, revocation by key rotation, and unlinkability
metrics suitable for controlled experiments.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class CancelableTemplate:
    values: np.ndarray
    key_id: str


def _rng_from_secret(secret: str) -> np.random.Generator:
    if not secret:
        raise ValueError("secret must be non-empty")
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    return np.random.default_rng(seed)


def protect_embedding(embedding, secret: str, output_dim: int | None = None) -> CancelableTemplate:
    x = np.asarray(embedding, dtype=np.float64).reshape(-1)
    if x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("embedding must contain finite values")
    d = int(output_dim or x.size)
    if d <= 0 or d > x.size:
        raise ValueError("output_dim must be in [1, embedding dimension]")
    rng = _rng_from_secret(secret)
    mat = rng.normal(size=(x.size, d))
    q, _ = np.linalg.qr(mat)
    projected = x @ q[:, :d]
    signs = rng.choice(np.array([-1.0, 1.0]), size=d)
    protected = projected * signs
    norm = np.linalg.norm(protected)
    if norm > 0:
        protected = protected / norm
    key_id = hashlib.sha256(("tapf-key:" + secret).encode("utf-8")).hexdigest()[:16]
    return CancelableTemplate(protected.astype(np.float32), key_id)


def cosine_similarity(a, b) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom else 0.0


def cross_key_unlinkability(embedding, secret_a: str, secret_b: str) -> float:
    """Return 1-|cosine|; larger values indicate stronger empirical cross-key separation."""
    ta = protect_embedding(embedding, secret_a)
    tb = protect_embedding(embedding, secret_b)
    return float(1.0 - abs(cosine_similarity(ta.values, tb.values)))


def revocation_check(embedding, old_secret: str, new_secret: str, max_cross_similarity: float = 0.35) -> dict:
    old = protect_embedding(embedding, old_secret)
    new = protect_embedding(embedding, new_secret)
    similarity = abs(cosine_similarity(old.values, new.values))
    return {
        "old_key_id": old.key_id,
        "new_key_id": new.key_id,
        "cross_key_similarity": float(similarity),
        "revocation_pass": bool(similarity <= max_cross_similarity and old.key_id != new.key_id),
        "scope": "Empirical keyed-transform check; not proof of irreversibility or unlinkability.",
    }
