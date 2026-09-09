"""Independent empirical privacy attacks for TAPF-MIN v3.

These attacks evaluate the exact released representation. They are empirical evidence,
not formal privacy guarantees. The module includes closed-set identity classification,
pairwise linkage verification, and repeated-release aggregation.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.run_cremad_final_validation import _attack, repeated_release_attack


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if den == 0 else float(np.dot(a, b) / den)


def pairwise_linkage_auc(Z, identities, *, seed=42, negatives_per_positive=3):
    """Same-person vs different-person linkage AUC using cosine similarity.

    Pairs are generated without using the task label. High AUC means a verifier can
    link two releases to the same identity. AUC near 0.5 is desired.
    """
    Z = np.asarray(Z, dtype=np.float32)
    identities = np.asarray(identities)
    if Z.ndim == 1:
        Z = Z[:, None]
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    by_id = {a: np.flatnonzero(identities == a) for a in np.unique(identities)}
    all_ids = list(by_id)
    for a, idx in by_id.items():
        if len(idx) < 2:
            continue
        order = idx.copy(); rng.shuffle(order)
        for i in range(len(order) - 1):
            positives.append(_cosine_similarity(Z[order[i]], Z[order[i+1]]))
            others = [x for x in all_ids if x != a]
            for _ in range(int(negatives_per_positive)):
                b = rng.choice(others)
                j = int(rng.choice(by_id[b]))
                negatives.append(_cosine_similarity(Z[order[i]], Z[j]))
    if not positives or not negatives:
        raise ValueError("insufficient repeated identities for linkage attack")
    y = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
    s = np.r_[positives, negatives]
    auc = float(roc_auc_score(y, s))
    effective = float(max(auc, 1.0 - auc))
    return {
        "attack": "pairwise_cosine_linkage",
        "auc": auc,
        "effective_auc": effective,
        "identity_advantage": float(effective - 0.5),
        "positive_pairs": len(positives),
        "negative_pairs": len(negatives),
    }


def nearest_centroid_linkage(Z, identities, *, seed=42):
    """Session-style linkage: enroll on half of each identity's releases, test on rest."""
    Z = np.asarray(Z, dtype=np.float32)
    identities = np.asarray(identities)
    rng = np.random.default_rng(seed)
    centroids = {}
    tests, labels = [], []
    for a in sorted(np.unique(identities)):
        idx = np.flatnonzero(identities == a).copy(); rng.shuffle(idx)
        if len(idx) < 2:
            continue
        cut = max(1, len(idx)//2)
        centroids[a] = Z[idx[:cut]].mean(axis=0)
        for j in idx[cut:]:
            tests.append(Z[j]); labels.append(a)
    if len(centroids) < 2 or not tests:
        raise ValueError("insufficient identities for centroid linkage")
    ids = list(centroids)
    pred = []
    for z in tests:
        scores = [_cosine_similarity(z, centroids[a]) for a in ids]
        pred.append(ids[int(np.argmax(scores))])
    labels = np.asarray(labels); pred = np.asarray(pred)
    acc = float(np.mean(labels == pred))
    chance = 1.0 / len(ids)
    return {
        "attack": "nearest_centroid_linkage",
        "accuracy": acc,
        "chance_accuracy": chance,
        "accuracy_advantage": max(0.0, acc - chance),
        "identities": len(ids),
        "test_releases": len(labels),
    }


def audit_representation(Z, identities, *, seed=42):
    """Run complementary attacks on one exact release representation."""
    return {
        "closed_set_identity": _attack(Z, identities, seed, bootstrap_n=400),
        "pairwise_linkage": pairwise_linkage_auc(Z, identities, seed=seed + 100),
        "centroid_linkage": nearest_centroid_linkage(Z, identities, seed=seed + 200),
        "repeated_release_identity": repeated_release_attack(Z, identities, seed + 300),
        "scope": "Empirical attacks only; no anonymity or DP claim follows from passing them.",
    }
