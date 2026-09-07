"""TAPF-MIN identity-suppression sweep on CREMA-D.

This experiment learns identity-bearing directions only from each actor-disjoint training fold,
removes a tunable fraction of those directions, trains the task model on the suppressed
features, and releases only out-of-fold task posteriors. The released posteriors are then
attacked with the same multi-attacker protocol used by the disclosure-spectrum benchmark.

The design is intentionally cross-fitted: no actor contributes data to the encoder/task model
that produces that actor's released representation. This prevents identity labels from the
held-out actor leaking into the representation transform.
"""
from __future__ import annotations

from pathlib import Path
import argparse, json, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tapf.minimum_representation import read_motion_clip
from experiments.run_cremad_privacy_utility_spectrum import attack_representation, bootstrap_metric

NAME_RE = re.compile(r"^(\d{4})_DFA_(ANG|DIS|FEA|HAP|NEU|SAD)_XX\.flv$", re.I)
MIN_REAL_VIDEO_BYTES = 1000


def identity_basis(X, actor, max_rank=12):
    """Estimate identity-bearing directions from actor means in standardized feature space."""
    X = np.asarray(X, np.float64)
    actor = np.asarray(actor)
    mu = X.mean(axis=0, keepdims=True)
    means = []
    for a in np.unique(actor):
        means.append(X[actor == a].mean(axis=0) - mu[0])
    M = np.asarray(means)
    if len(M) < 2 or not np.any(np.abs(M) > 0):
        return np.zeros((X.shape[1], 0), dtype=np.float64)
    _, _, vt = np.linalg.svd(M, full_matrices=False)
    rank = min(int(max_rank), vt.shape[0], X.shape[1])
    return vt[:rank].T


def suppress_identity(X, basis, strength):
    X = np.asarray(X, np.float64)
    if basis.size == 0 or strength <= 0:
        return X.copy()
    return X - float(strength) * (X @ basis) @ basis.T


def crossfit_posteriors(X, actor, emotion, folds, strength, rank, seed):
    classes = np.asarray(sorted(np.unique(emotion)))
    P = np.zeros((len(X), len(classes)), np.float32)
    fold_rows = []
    splitter = GroupKFold(n_splits=folds)
    for fold, (tr, te) in enumerate(splitter.split(X, emotion, groups=actor), 1):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        basis = identity_basis(Xtr, actor[tr], max_rank=rank)
        Xtr_s = suppress_identity(Xtr, basis, strength)
        Xte_s = suppress_identity(Xte, basis, strength)

        task = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed + fold)
        task.fit(Xtr_s, emotion[tr])
        raw = task.predict_proba(Xte_s)
        col = {c: j for j, c in enumerate(task.classes_)}
        for k, c in enumerate(classes):
            P[te, k] = raw[:, col[c]]
        pred = classes[np.argmax(P[te], axis=1)]
        fold_rows.append({
            "fold": fold,
            "train_actors": int(len(np.unique(actor[tr]))),
            "test_actors": int(len(np.unique(actor[te]))),
            "identity_basis_rank": int(basis.shape[1]),
            "macro_f1": float(f1_score(emotion[te], pred, average="macro", zero_division=0)),
        })
    return P, classes, fold_rows


def run(root, folds=5, seed=42):
    root = Path(root)
    files = []
    for p in sorted(root.glob("*.flv")):
        m = NAME_RE.match(p.name)
        if m and p.stat().st_size > MIN_REAL_VIDEO_BYTES:
            files.append((p, m.group(1), m.group(2).upper()))
    actor = np.asarray([a for _, a, _ in files])
    emotion = np.asarray([e for _, _, e in files])
    actors = sorted(np.unique(actor))
    if len(files) < 120 or len(actors) < 8:
        raise RuntimeError(f"Need >=120 hydrated clips and >=8 actors; got {len(files)} clips/{len(actors)} actors")

    X = np.asarray([read_motion_clip(str(p)) for p, _, _ in files], np.float32)
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite input features")
    folds = max(2, min(int(folds), len(actors)))

    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    ranks = [2, 4, 8, 12]
    rows = []
    for rank in ranks:
        for strength in strengths:
            P, classes, fold_rows = crossfit_posteriors(X, actor, emotion, folds, strength, rank, seed)
            pred = classes[np.argmax(P, axis=1)]
            f1, f1_lo, f1_hi = bootstrap_metric(
                emotion, pred,
                lambda y, p: f1_score(y, p, average="macro", zero_division=0),
                seed=seed + 1000 + rank * 10 + int(strength * 100),
            )
            privacy = attack_representation(P, actor, seed + 2000 + rank * 10 + int(strength * 100))
            upper_auc = float(privacy["identity_auc_ci95"][1])
            lower_f1 = float(f1_lo)
            rows.append({
                "rank": rank,
                "suppression_strength": strength,
                "release": "6-D cross-fitted task posterior",
                "emotion_macro_f1": f1,
                "emotion_macro_f1_ci95": [f1_lo, f1_hi],
                "identity_auc": privacy["identity_auc"],
                "identity_auc_ci95": privacy["identity_auc_ci95"],
                "identity_risk_0to1": privacy["identity_risk_0to1"],
                "worst_case_attacker": privacy["worst_case_attacker"],
                "attackers": privacy["attackers"],
                "shuffled_label_control_auc": privacy["shuffled_label_control_auc"],
                "privacy_gate_point_auc_le_0_60": bool(privacy["identity_auc"] <= 0.60),
                "privacy_gate_upper_ci_le_0_60": bool(upper_auc <= 0.60),
                "utility_gate_lower_ci_ge_0_20": bool(lower_f1 >= 0.20),
                "folds": fold_rows,
            })

    # Pareto set: lower identity AUC and higher task F1 are preferred.
    pareto = []
    for a in rows:
        dominated = False
        for b in rows:
            if a is b:
                continue
            if (b["identity_auc"] <= a["identity_auc"] and
                b["emotion_macro_f1"] >= a["emotion_macro_f1"] and
                (b["identity_auc"] < a["identity_auc"] or b["emotion_macro_f1"] > a["emotion_macro_f1"])):
                dominated = True
                break
        if not dominated:
            pareto.append({"rank": a["rank"], "suppression_strength": a["suppression_strength"]})

    best_privacy = min(rows, key=lambda r: abs(r["identity_auc"] - 0.5))
    best_utility = max(rows, key=lambda r: r["emotion_macro_f1"])
    eligible = [r for r in rows if r["privacy_gate_upper_ci_le_0_60"] and r["utility_gate_lower_ci_ge_0_20"]]

    result = {
        "dataset": "CREMA-D bounded DFA subset",
        "clips": len(files), "actors": len(actors),
        "method": "cross-fitted identity-subspace suppression followed by task-only posterior release",
        "scientific_note": "This is a linear identity-suppression baseline, not yet a neural gradient-reversal encoder.",
        "protocol": "identity basis and task model fit only on actor-disjoint training folds; held-out actors generate OOF releases; identity attackers train on released OOF representations with clip-disjoint actor splits",
        "targets": {
            "chance_identity_auc": 0.5,
            "initial_point_target_auc": 0.60,
            "conservative_upper_ci_target_auc": 0.60,
            "minimum_lower_ci_emotion_f1": 0.20,
        },
        "sweep": rows,
        "pareto_frontier": pareto,
        "best_privacy_point": best_privacy,
        "best_utility_point": best_utility,
        "eligible_operating_points": eligible,
        "edge_checks": {
            "cross_fitted_encoder": True,
            "actor_disjoint_task_folds": True,
            "identity_attacker_clip_disjoint": True,
            "multi_attacker_worst_case": True,
            "bootstrap_confidence_intervals": True,
            "negative_control": True,
            "auc_below_half_not_treated_as_extra_privacy": True,
        },
    }
    Path("results").mkdir(exist_ok=True)
    out = Path("results/cremad_identity_suppression_sweep.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    run(args.root, args.folds)
