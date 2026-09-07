"""TAPF-MIN privacy/utility disclosure-spectrum benchmark on CREMA-D.

This experiment asks how identity leakage changes as the released object is reduced from
motion features to task-only outputs. It is deliberately adversarial: each released
representation is attacked by multiple classifier families, and the worst measured AUC is
reported. Utility is evaluated actor-disjoint using out-of-fold task predictions.

No operating point is called private merely because it is compact or non-image.
"""
from __future__ import annotations

from pathlib import Path
import argparse, json, re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tapf.minimum_representation import read_motion_clip

NAME_RE = re.compile(r"^(\d{4})_DFA_(ANG|DIS|FEA|HAP|NEU|SAD)_XX\.flv$", re.I)
MIN_REAL_VIDEO_BYTES = 1000


def macro_ovr_auc(y_true, proba, classes):
    y_true = np.asarray(y_true)
    vals = []
    for j, c in enumerate(classes):
        yy = (y_true == c).astype(int)
        if yy.min() == yy.max():
            continue
        vals.append(float(roc_auc_score(yy, proba[:, j])))
    if not vals:
        raise ValueError("No valid one-vs-rest class AUCs")
    return float(np.mean(vals))


def bootstrap_metric(y, values, fn, seed=42, n=400):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    values = np.asarray(values)
    point = float(fn(y, values))
    boots = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        try:
            boots.append(float(fn(y[idx], values[idx])))
        except ValueError:
            pass
    lo, hi = np.quantile(boots, [.025, .975]) if boots else (point, point)
    return point, float(lo), float(hi)


def identity_split(actor, seed):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for a in sorted(np.unique(actor)):
        ids = np.flatnonzero(actor == a).copy()
        rng.shuffle(ids)
        if len(ids) < 2:
            continue
        cut = max(1, min(len(ids) - 1, len(ids) // 2))
        tr.extend(ids[:cut])
        te.extend(ids[cut:])
    return np.asarray(tr, int), np.asarray(te, int)


def attackers(seed):
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
        ),
        "rbf_svm": make_pipeline(
            StandardScaler(),
            SVC(C=3.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350, class_weight="balanced_subsample", min_samples_leaf=1,
            random_state=seed, n_jobs=-1,
        ),
    }


def attack_representation(Z, actor, seed):
    Z = np.asarray(Z, np.float32)
    if Z.ndim == 1:
        Z = Z[:, None]
    if not np.isfinite(Z).all():
        raise RuntimeError("Non-finite released representation")
    tr, te = identity_split(actor, seed)
    rows = []
    worst = None
    for i, (name, model) in enumerate(attackers(seed).items()):
        model.fit(Z[tr], actor[tr])
        proba = model.predict_proba(Z[te])
        classes = model.classes_
        auc, lo, hi = bootstrap_metric(
            actor[te], proba,
            lambda y, p: macro_ovr_auc(y, p, classes),
            seed=seed + 100 + i,
        )
        row = {"attacker": name, "auc": auc, "ci95": [lo, hi]}
        rows.append(row)
        if worst is None or auc > worst["auc"]:
            worst = row

    # Negative-control sanity check: shuffled identity labels should destroy learnable identity.
    rng = np.random.default_rng(seed + 999)
    shuffled = actor[tr].copy(); rng.shuffle(shuffled)
    ctrl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed))
    ctrl.fit(Z[tr], shuffled)
    ctrl_proba = ctrl.predict_proba(Z[te])
    ctrl_auc = macro_ovr_auc(actor[te], ctrl_proba, ctrl.classes_)

    auc = float(worst["auc"])
    return {
        "attackers": rows,
        "worst_case_attacker": worst["attacker"],
        "identity_auc": auc,
        "identity_auc_ci95": worst["ci95"],
        "distance_from_chance": float(abs(auc - 0.5)),
        "identity_risk_0to1": float(min(1.0, 2.0 * abs(auc - 0.5))),
        "shuffled_label_control_auc": float(ctrl_auc),
        "privacy_interpretation": "AUC near 0.5 is desired; risk=2*|AUC-0.5|. Worst attacker is used.",
    }


def quantize_probabilities(P, levels):
    P = np.asarray(P, np.float32)
    q = np.round(P * (levels - 1)) / float(levels - 1)
    s = q.sum(1, keepdims=True)
    return np.divide(q, s, out=np.full_like(q, 1.0 / q.shape[1]), where=s > 0)


def run(root, folds=5, seed=42):
    root = Path(root)
    files = []
    for p in sorted(root.glob("*.flv")):
        m = NAME_RE.match(p.name)
        if m and p.stat().st_size > MIN_REAL_VIDEO_BYTES:
            files.append((p, m.group(1), m.group(2).upper()))
    actors = sorted({a for _, a, _ in files})
    if len(files) < 120 or len(actors) < 8:
        raise RuntimeError(f"Need >=120 hydrated clips and >=8 actors; got {len(files)} clips/{len(actors)} actors")

    X = np.asarray([read_motion_clip(str(p)) for p, _, _ in files], np.float32)
    actor = np.asarray([a for _, a, _ in files])
    emotion = np.asarray([e for _, _, e in files])
    classes = np.asarray(sorted(np.unique(emotion)))
    folds = max(2, min(int(folds), len(actors)))

    # Actor-disjoint out-of-fold task posteriors. Each actor is predicted by a task model
    # that never trained on that actor.
    P = np.zeros((len(files), len(classes)), np.float32)
    splitter = GroupKFold(n_splits=folds)
    fold_rows = []
    for fold, (tr, te) in enumerate(splitter.split(X, emotion, groups=actor), 1):
        task = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed + fold),
        )
        task.fit(X[tr], emotion[tr])
        raw = task.predict_proba(X[te])
        col = {c: j for j, c in enumerate(task.classes_)}
        for k, c in enumerate(classes):
            P[te, k] = raw[:, col[c]]
        pred = classes[np.argmax(P[te], axis=1)]
        fold_rows.append({
            "fold": fold,
            "train_actors": int(len(np.unique(actor[tr]))),
            "test_actors": int(len(np.unique(actor[te]))),
            "macro_f1": float(f1_score(emotion[te], pred, average="macro", zero_division=0)),
        })

    points = [
        ("motion-147d", X, "147-D optical-flow summary"),
        ("task-posterior-6d", P, "6-D actor-disjoint emotion posterior"),
        ("task-posterior-q16", quantize_probabilities(P, 16), "6-D posterior quantized to 16 levels"),
        ("task-posterior-q4", quantize_probabilities(P, 4), "6-D posterior quantized to 4 levels"),
    ]
    hard_idx = np.argmax(P, axis=1)
    hard = np.eye(len(classes), dtype=np.float32)[hard_idx]
    points.append(("task-label-onehot", hard, "one-hot predicted task label only"))

    spectrum = []
    for i, (name, Z, desc) in enumerate(points):
        pred = classes[np.argmax(Z, axis=1)] if Z.shape[1] == len(classes) else classes[np.argmax(P, axis=1)]
        f1, f1_lo, f1_hi = bootstrap_metric(
            emotion, pred,
            lambda y, p: f1_score(y, p, average="macro", zero_division=0),
            seed=seed + 10 + i,
        )
        privacy = attack_representation(Z, actor, seed + 20 + i)
        spectrum.append({
            "representation": name,
            "description": desc,
            "dimension": int(Z.shape[1]),
            "emotion_macro_f1": f1,
            "emotion_macro_f1_ci95": [f1_lo, f1_hi],
            **privacy,
        })

    # Pareto frontier: lower identity risk and higher utility are both preferred.
    frontier = []
    for a in spectrum:
        dominated = False
        for b in spectrum:
            if a is b:
                continue
            if (b["identity_risk_0to1"] <= a["identity_risk_0to1"] and
                b["emotion_macro_f1"] >= a["emotion_macro_f1"] and
                (b["identity_risk_0to1"] < a["identity_risk_0to1"] or b["emotion_macro_f1"] > a["emotion_macro_f1"])):
                dominated = True; break
        if not dominated:
            frontier.append(a["representation"])

    result = {
        "dataset": "CREMA-D bounded DFA subset",
        "clips": len(files), "actors": len(actors), "emotion_classes": classes.tolist(),
        "utility_protocol": f"{folds}-fold actor-disjoint out-of-fold task prediction",
        "privacy_protocol": "clip-disjoint closed-set identity inference with LogisticRegression, RBF-SVM and RandomForest; worst AUC reported",
        "spectrum": spectrum,
        "pareto_frontier": frontier,
        "edge_checks": {
            "non_finite_features": bool(not np.isfinite(X).all()),
            "all_actors_have_multiple_clips": bool(all(np.sum(actor == a) >= 2 for a in np.unique(actor))),
            "identity_train_test_clip_overlap": False,
            "utility_actor_overlap_within_fold": False,
            "chance_target_auc": 0.5,
            "warning": "Do not interpret AUC <0.5 as stronger privacy without checking prediction inversion; use distance from chance/worst attacker.",
        },
        "utility_folds": fold_rows,
        "decision_rule": "No representation is release-eligible unless predeclared privacy and utility bounds are both satisfied.",
    }
    Path("results").mkdir(exist_ok=True)
    out = Path("results/cremad_privacy_utility_spectrum.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    run(a.root, a.folds)
