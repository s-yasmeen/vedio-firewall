"""Clean fixed-model FER/privacy audit for TAPF-MIN v3.

Purpose
-------
The original OOF posterior stress test transforms different actor folds with different
FER checkpoints. That is useful as a stress diagnostic but confounds biometric leakage
with fold/model fingerprints. This experiment removes that confound:

1. split actors into FER-development actors and privacy-audit actors;
2. train exactly one FER checkpoint using development actors only;
3. freeze it;
4. transform every audit actor with that same checkpoint;
5. evaluate FER utility and identity/linkage attacks only on the held-out audit actors.

This is still a controlled CREMA-D development protocol, not clinical validation.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

from experiments.run_cremad_final_validation import _attack, repeated_release_attack
from experiments.run_cremad_v3_deep_fer import (
    EMOTIONS,
    MIN_REAL_VIDEO_BYTES,
    NAME_RE,
    seed_all,
    train_one_fold,
)


def actor_cluster_bootstrap_f1(y_true, y_pred, actors, *, seed=42, n=1000):
    """Bootstrap task utility by resampling actors, not individual clips."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    actors = np.asarray(actors)
    unique = np.asarray(sorted(np.unique(actors)))
    point = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        ys, ps = [], []
        for a in sampled:
            idx = np.flatnonzero(actors == a)
            ys.extend(y_true[idx].tolist())
            ps.extend(y_pred[idx].tolist())
        vals.append(float(f1_score(ys, ps, average="macro", zero_division=0)))
    return point, float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def split_actor_cohorts(actors, *, seed=42, audit_fraction=0.33, min_audit_actors=8):
    unique = np.asarray(sorted(np.unique(actors)))
    if len(unique) < min_audit_actors + 4:
        raise ValueError("Need more actors for separate FER-development and privacy-audit cohorts")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy(); rng.shuffle(shuffled)
    n_audit = max(int(min_audit_actors), int(round(len(unique) * float(audit_fraction))))
    n_audit = min(n_audit, len(unique) - 4)
    audit = set(shuffled[:n_audit].tolist())
    develop = set(shuffled[n_audit:].tolist())
    return develop, audit


def run(
    root,
    *,
    epochs=8,
    batch_size=8,
    frames_per_clip=12,
    image_size=112,
    seed=42,
    pretrained=True,
    audit_fraction=0.33,
):
    seed_all(seed)
    root = Path(root)
    raw = []
    for p in sorted(root.glob("*.flv")):
        m = NAME_RE.match(p.name)
        if m and p.stat().st_size > MIN_REAL_VIDEO_BYTES:
            raw.append((p, m.group(1), m.group(2).upper()))
    if len(raw) < 120:
        raise RuntimeError(f"Need >=120 real clips; got {len(raw)}")

    actors_all = np.asarray([a for _, a, _ in raw])
    dev_actors, audit_actors = split_actor_cohorts(
        actors_all, seed=seed, audit_fraction=audit_fraction
    )
    records = [(p, a, e, i) for i, (p, a, e) in enumerate(raw)]
    train_records = [r for r in records if r[1] in dev_actors]
    audit_records = [r for r in records if r[1] in audit_actors]

    if set(dev_actors).intersection(audit_actors):
        raise RuntimeError("Actor leakage between development and audit cohorts")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P_raw, original_ids, history, config = train_one_fold(
        train_records,
        audit_records,
        epochs=epochs,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        image_size=image_size,
        seed=seed + 101,
        pretrained=pretrained,
        device=device,
    )

    by_id = {int(i): P_raw[j] for j, i in enumerate(original_ids)}
    audit_ids = np.asarray([int(r[3]) for r in audit_records])
    P = np.asarray([by_id[int(i)] for i in audit_ids], dtype=np.float32)
    audit_actor = np.asarray([r[1] for r in audit_records])
    audit_emotion = np.asarray([r[2] for r in audit_records])
    pred = EMOTIONS[np.argmax(P, axis=1)]

    f1, f1_lo, f1_hi = actor_cluster_bootstrap_f1(
        audit_emotion, pred, audit_actor, seed=seed + 700, n=1000
    )
    accuracy = float(accuracy_score(audit_emotion, pred))

    # All rows below come from the SAME frozen FER checkpoint and identities absent
    # from FER training. This removes the fold-checkpoint fingerprint confound.
    identity = _attack(P, audit_actor, seed + 900, bootstrap_n=400)
    repeated = repeated_release_attack(P, audit_actor, seed + 1200)

    result = {
        "dataset": "CREMA-D DFA cohort supplied to runner",
        "protocol": "single frozen FER checkpoint; held-out privacy-audit identities",
        "scientific_scope": "Controlled development audit; not clinical validation.",
        "clips_total": len(records),
        "development_actors": len(dev_actors),
        "privacy_audit_actors": len(audit_actors),
        "actor_overlap": len(set(dev_actors).intersection(audit_actors)),
        "same_frozen_model_for_all_audit_actors": True,
        "local_fer_on_privacy_audit_cohort": {
            "accuracy": accuracy,
            "macro_f1": f1,
            "actor_cluster_macro_f1_ci95": [f1_lo, f1_hi],
        },
        "empirical_privacy_of_nonprivate_posterior": {
            "clip_identity": identity,
            "repeated_release_identity": repeated,
            "interpretation": "Empirical held-out-identity attack on one frozen task model; not a formal privacy theorem.",
        },
        "training_history": history,
        "model_config": config,
        "training_parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "frames_per_clip": frames_per_clip,
            "image_size": image_size,
            "pretrained_imagenet": bool(pretrained),
            "seed": seed,
            "audit_fraction": float(audit_fraction),
        },
        "remaining_limitations": [
            "privacy-audit sample size may still be small on the bounded workflow cohort",
            "identity-attack CI implementation is not yet hierarchical by identity",
            "CREMA-D acted emotion is not telemedicine/clinical validation",
        ],
    }
    Path("results").mkdir(exist_ok=True)
    out = Path("results/cremad_v3_fixed_model_audit.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--frames-per-clip", type=int, default=12)
    ap.add_argument("--image-size", type=int, default=112)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audit-fraction", type=float, default=0.33)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()
    run(
        args.root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        frames_per_clip=args.frames_per_clip,
        image_size=args.image_size,
        seed=args.seed,
        pretrained=not args.no_pretrained,
        audit_fraction=args.audit_fraction,
    )
