import numpy as np
from attackers.sequence_attack import aggregate_embeddings, cosine_similarity, cumulative_sequence_curve


def test_sequence_aggregation_strengthens_consistent_signal():
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    frames = [
        np.array([0.9, 0.1, 0.0], dtype=np.float32),
        np.array([0.8, -0.1, 0.0], dtype=np.float32),
        np.array([0.95, 0.05, 0.0], dtype=np.float32),
    ]
    pooled = aggregate_embeddings(frames)
    assert pooled is not None
    assert cosine_similarity(ref, pooled) > 0.95


def test_cumulative_curve_length_matches_frames():
    ref = np.array([1.0, 0.0], dtype=np.float32)
    frames = [np.array([1.0, 0.0]), None, np.array([0.8, 0.2])]
    curve = cumulative_sequence_curve(ref, frames)
    assert len(curve) == len(frames)
    assert np.isfinite(curve[-1])
