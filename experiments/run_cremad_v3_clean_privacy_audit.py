"""TAPF-MIN v3 clean fixed-model privacy suite.

One FER checkpoint is trained on development actors, frozen, and then used for every
privacy-audit actor. The exact posterior is attacked by multiple complementary linkage
attacks. This removes the fold-checkpoint fingerprint confound in older OOF diagnostics.
"""
from __future__ import annotations

from pathlib import Path
import argparse, json
import numpy as np
from sklearn.metrics import accuracy_score
import torch

from experiments.run_cremad_v3_deep_fer import (
    EMOTIONS, MIN_REAL_VIDEO_BYTES, NAME_RE, seed_all, train_one_fold,
)
from experiments.run_cremad_v3_fixed_model_audit import (
    actor_cluster_bootstrap_f1, split_actor_cohorts,
)
from tapf.privacy_attacks_v3 import audit_representation


def run(root, *, epochs=8, batch_size=8, frames_per_clip=12, image_size=112,
        seed=42, pretrained=True, audit_fraction=.33):
    seed_all(seed)
    root = Path(root)
    raw = []
    for p in sorted(root.glob("*.flv")):
        m = NAME_RE.match(p.name)
        if m and p.stat().st_size > MIN_REAL_VIDEO_BYTES:
            raw.append((p, m.group(1), m.group(2).upper()))
    if len(raw) < 120:
        raise RuntimeError(f"Need >=120 real clips; got {len(raw)}")

    all_actor = np.asarray([a for _, a, _ in raw])
    dev_actors, audit_actors = split_actor_cohorts(
        all_actor, seed=seed, audit_fraction=audit_fraction
    )
    if dev_actors & audit_actors:
        raise RuntimeError("development/privacy audit actor overlap")

    records = [(p, a, e, i) for i, (p, a, e) in enumerate(raw)]
    train_records = [r for r in records if r[1] in dev_actors]
    audit_records = [r for r in records if r[1] in audit_actors]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P_raw, ids, history, config = train_one_fold(
        train_records, audit_records, epochs=epochs, batch_size=batch_size,
        frames_per_clip=frames_per_clip, image_size=image_size, seed=seed+101,
        pretrained=pretrained, device=device,
    )
    by_id = {int(i): P_raw[j] for j, i in enumerate(ids)}
    audit_ids = np.asarray([r[3] for r in audit_records], int)
    P = np.asarray([by_id[int(i)] for i in audit_ids], np.float32)
    actor = np.asarray([r[1] for r in audit_records])
    emotion = np.asarray([r[2] for r in audit_records])
    pred = EMOTIONS[np.argmax(P, axis=1)]
    f1, lo, hi = actor_cluster_bootstrap_f1(emotion, pred, actor, seed=seed+700, n=1000)

    attacks = audit_representation(P, actor, seed=seed+900)
    result = {
        "schema": "tapf-min-clean-privacy-audit/v1",
        "dataset": "CREMA-D DFA cohort supplied to runner",
        "scientific_scope": "controlled actor-disjoint development audit; not clinical validation",
        "protocol": {
            "fer_training_actors": len(dev_actors),
            "privacy_audit_actors": len(audit_actors),
            "actor_overlap": 0,
            "single_frozen_fer_checkpoint_for_audit": True,
            "identity_attacker_training_does_not_update_fer": True,
        },
        "utility": {
            "accuracy": float(accuracy_score(emotion, pred)),
            "macro_f1": float(f1),
            "actor_cluster_macro_f1_ci95": [float(lo), float(hi)],
        },
        "empirical_privacy": attacks,
        "training_history": history,
        "model_config": config,
        "parameters": {
            "epochs": epochs, "batch_size": batch_size,
            "frames_per_clip": frames_per_clip, "image_size": image_size,
            "seed": seed, "pretrained": bool(pretrained),
            "audit_fraction": float(audit_fraction),
        },
    }
    Path("results").mkdir(exist_ok=True)
    out = Path("results/cremad_v3_clean_privacy_audit.json")
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
    ap.add_argument("--audit-fraction", type=float, default=.33)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    run(a.root, epochs=a.epochs, batch_size=a.batch_size,
        frames_per_clip=a.frames_per_clip, image_size=a.image_size,
        seed=a.seed, pretrained=not a.no_pretrained, audit_fraction=a.audit_fraction)
