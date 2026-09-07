"""Sequence-level identity leakage utilities.

These functions aggregate embeddings across frames to test whether weak residual identity
signals become stronger over time. This is an empirical attack protocol, not a formal privacy bound.
"""
from __future__ import annotations
import numpy as np


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def aggregate_embeddings(embeddings, mode: str = "mean"):
    valid = [normalize(e) for e in embeddings if e is not None]
    if not valid:
        return None
    mat = np.vstack(valid)
    if mode == "mean":
        return normalize(mat.mean(axis=0))
    if mode == "median":
        return normalize(np.median(mat, axis=0))
    raise ValueError(f"Unknown aggregation mode: {mode}")


def cosine_similarity(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    a, b = normalize(a), normalize(b)
    return float(np.dot(a, b))


def sequence_similarity(reference_embedding, frame_embeddings, mode: str = "mean") -> float:
    pooled = aggregate_embeddings(frame_embeddings, mode=mode)
    return cosine_similarity(reference_embedding, pooled)


def cumulative_sequence_curve(reference_embedding, frame_embeddings):
    """Return similarity after aggregating 1..N frames."""
    curve = []
    running = []
    for emb in frame_embeddings:
        if emb is not None:
            running.append(emb)
        curve.append(sequence_similarity(reference_embedding, running))
    return curve
