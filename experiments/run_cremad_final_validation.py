"""Consolidated final-validation benchmark for TAPF-MIN.

v2.2 safety protocol (predeclared for new runs):
- actor-disjoint task folds,
- multiple minimum-disclosure releases,
- four independent post-hoc identity attackers including an MLP,
- repeated-release aggregation attack,
- bootstrap confidence intervals,
- chance-centered privacy metric that treats AUC above or below 0.50 symmetrically,
- conservative fail-closed gate.

Scientific scope: controlled CREMA-D emotion benchmark on the validated 147-D
motion descriptor. It is not clinical validation and does not prove anonymity.

Important: AUC < 0.50 is not automatically safer because an attacker can invert
its scores. Privacy is therefore measured as distance from chance:
    identity_advantage = |AUC - 0.50|
and equivalently:
    effective_auc = 0.50 + identity_advantage = max(AUC, 1-AUC).
The v2.2 target is identity_advantage <= 0.05 (effective AUC <= 0.55).
"""
from __future__ import annotations

from pathlib import Path
import argparse, json, re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from tapf.minimum_representation import read_motion_clip
from experiments.run_cremad_v21_adversarial import EMOTIONS, crossfit, quantize_probabilities
from experiments.run_cremad_privacy_utility_spectrum import bootstrap_metric, macro_ovr_auc, identity_split

NAME_RE = re.compile(r"^(\d{4})_DFA_(ANG|DIS|FEA|HAP|NEU|SAD)_XX\.flv$", re.I)
MIN_REAL_VIDEO_BYTES = 1000
CHANCE_AUC = 0.50
MAX_IDENTITY_ADVANTAGE = 0.05
MAX_EFFECTIVE_AUC = CHANCE_AUC + MAX_IDENTITY_ADVANTAGE  # 0.55
MIN_UTILITY_LOWER_F1 = 0.20


def identity_advantage(auc: float) -> float:
    return float(abs(float(auc) - CHANCE_AUC))


def effective_auc(auc: float) -> float:
    """Invertible attacker performance; values below 0.5 are reflected above chance."""
    return float(CHANCE_AUC + identity_advantage(auc))


def conservative_advantage_upper(ci95) -> float:
    """Largest distance from chance contained in a 95% AUC confidence interval."""
    lo, hi = map(float, ci95)
    return float(max(abs(lo - CHANCE_AUC), abs(hi - CHANCE_AUC)))


def attacker_models(seed: int):
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed)),
        "rbf_svm": make_pipeline(StandardScaler(), SVC(
            C=3.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        "random_forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=1,
            random_state=seed, n_jobs=-1),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
            max_iter=1200, early_stopping=False, random_state=seed)),
    }


def _attack(Z, actor, seed: int, bootstrap_n: int = 300):
    Z = np.asarray(Z, np.float32)
    if Z.ndim == 1:
        Z = Z[:, None]
    tr, te = identity_split(actor, seed)
    rows = []
    worst = None
    for i, (name, model) in enumerate(attacker_models(seed).items()):
        model.fit(Z[tr], actor[tr])
        proba = model.predict_proba(Z[te])
        classes = model.classes_
        auc, lo, hi = bootstrap_metric(
            actor[te], proba, lambda y, p: macro_ovr_auc(y, p, classes),
            seed=seed + 100 + i, n=bootstrap_n)
        ci = [float(lo), float(hi)]
        adv = identity_advantage(auc)
        eff = effective_auc(auc)
        adv_upper = conservative_advantage_upper(ci)
        row = {
            "attacker": name,
            "auc": float(auc),
            "ci95": ci,
            "identity_advantage": adv,
            "effective_auc": eff,
            "advantage_upper_ci95": adv_upper,
            "effective_auc_upper_ci95": float(CHANCE_AUC + adv_upper),
        }
        rows.append(row)
        if worst is None or row["effective_auc"] > worst["effective_auc"]:
            worst = row
    return {
        "attackers": rows,
        "worst_case_attacker": worst["attacker"],
        "identity_auc": worst["auc"],
        "identity_auc_ci95": worst["ci95"],
        "identity_advantage": worst["identity_advantage"],
        "effective_auc": worst["effective_auc"],
        "advantage_upper_ci95": worst["advantage_upper_ci95"],
        "effective_auc_upper_ci95": worst["effective_auc_upper_ci95"],
        "identity_risk_0to1": float(min(1.0, 2 * worst["identity_advantage"])),
    }


def repeated_release_attack(Z, actor, seed: int):
    """Aggregate disjoint repeated releases before identity inference."""
    rng = np.random.default_rng(seed)
    ztr, ytr, zte, yte = [], [], [], []
    for a in sorted(np.unique(actor)):
        ids = np.flatnonzero(actor == a).copy()
        rng.shuffle(ids)
        if len(ids) < 4:
            continue
        cut = len(ids) // 2
        left, right = ids[:cut], ids[cut:]
        ztr.append(np.mean(Z[left], axis=0)); ytr.append(a)
        zte.append(np.mean(Z[right], axis=0)); yte.append(a)
    ztr = np.asarray(ztr, np.float32); zte = np.asarray(zte, np.float32)
    ytr = np.asarray(ytr); yte = np.asarray(yte)
    rows = []
    worst = None
    for name, model in attacker_models(seed + 50).items():
        model.fit(ztr, ytr)
        proba = model.predict_proba(zte)
        classes = model.classes_
        auc = float(macro_ovr_auc(yte, proba, classes))
        adv = identity_advantage(auc)
        eff = effective_auc(auc)
        row = {
            "attacker": name,
            "auc": auc,
            "identity_advantage": adv,
            "effective_auc": eff,
        }
        rows.append(row)
        if worst is None or row["effective_auc"] > worst["effective_auc"]:
            worst = row
    return {
        "aggregation": "mean of disjoint repeated releases per actor",
        "groups_per_actor": 2,
        "clips_per_group_approx": 3,
        "attackers": rows,
        "worst_case_attacker": worst["attacker"],
        "identity_auc": worst["auc"],
        "identity_advantage": worst["identity_advantage"],
        "effective_auc": worst["effective_auc"],
        "identity_risk_0to1": float(min(1.0, 2 * worst["identity_advantage"])),
    }


def evaluate_release(Z, actor, emotion, seed: int, label: str):
    pred = EMOTIONS[np.argmax(Z, axis=1)]
    f1, lo, hi = bootstrap_metric(
        emotion, pred, lambda y, p: f1_score(y, p, average="macro", zero_division=0),
        seed=seed, n=400)
    privacy = _attack(Z, actor, seed + 500)
    temporal = repeated_release_attack(Z, actor, seed + 900)
    point_privacy = bool(privacy["identity_advantage"] <= MAX_IDENTITY_ADVANTAGE)
    conservative_privacy = bool(privacy["advantage_upper_ci95"] <= MAX_IDENTITY_ADVANTAGE)
    temporal_privacy = bool(temporal["identity_advantage"] <= MAX_IDENTITY_ADVANTAGE)
    utility_ok = bool(lo >= MIN_UTILITY_LOWER_F1)
    return {
        "release": label,
        "dimension": int(Z.shape[1]),
        "emotion_macro_f1": float(f1),
        "emotion_macro_f1_ci95": [float(lo), float(hi)],
        "clip_identity": privacy,
        "repeated_release_identity": temporal,
        "privacy_gate_point_advantage_le_0_05": point_privacy,
        "privacy_gate_conservative_advantage_le_0_05": conservative_privacy,
        "temporal_gate_advantage_le_0_05": temporal_privacy,
        "utility_gate_lower_ci_ge_0_20": utility_ok,
    }


def run(root, folds=5, seed=42):
    root = Path(root)
    files = []
    for p in sorted(root.glob("*.flv")):
        m = NAME_RE.match(p.name)
        if m and p.stat().st_size > MIN_REAL_VIDEO_BYTES:
            files.append((p, m.group(1), m.group(2).upper()))
    actor = np.asarray([a for _, a, _ in files])
    emotion = np.asarray([e for _, _, e in files])
    if len(files) < 120 or len(np.unique(actor)) < 8:
        raise RuntimeError(f"Insufficient hydrated CREMA-D subset: {len(files)} clips/{len(np.unique(actor))} actors")
    X = np.asarray([read_motion_clip(str(p)) for p, _, _ in files], np.float32)
    folds = max(2, min(int(folds), len(np.unique(actor))))

    strengths = [0.0, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50]
    rows = []
    for i, s in enumerate(strengths):
        P, fold_rows = crossfit(X, actor, emotion, folds, s, seed + 1000 * i)
        reps = [
            ("posterior-6d", P),
            ("posterior-q8", quantize_probabilities(P, 8)),
            ("posterior-q4", quantize_probabilities(P, 4)),
            ("posterior-q2", quantize_probabilities(P, 2)),
            ("task-label-onehot", np.eye(len(EMOTIONS), dtype=np.float32)[np.argmax(P, axis=1)]),
        ]
        evals = []
        for j, (name, Z) in enumerate(reps):
            evals.append(evaluate_release(Z, actor, emotion, seed + 10000 * i + 100 * j, name))
        rows.append({"adversarial_strength": s, "folds": fold_rows, "releases": evals})

    candidates = []
    for r in rows:
        for e in r["releases"]:
            c = dict(e)
            c["adversarial_strength"] = r["adversarial_strength"]
            c["conservative_release_eligible"] = bool(
                c["privacy_gate_conservative_advantage_le_0_05"]
                and c["temporal_gate_advantage_le_0_05"]
                and c["utility_gate_lower_ci_ge_0_20"]
            )
            candidates.append(c)

    eligible = [c for c in candidates if c["conservative_release_eligible"]]
    best_privacy = min(candidates, key=lambda c: c["clip_identity"]["identity_advantage"])
    best_utility = max(candidates, key=lambda c: c["emotion_macro_f1"])
    best_joint = min(candidates, key=lambda c: (
        max(0.0, c["clip_identity"]["identity_advantage"] - MAX_IDENTITY_ADVANTAGE)
        + max(0.0, c["repeated_release_identity"]["identity_advantage"] - MAX_IDENTITY_ADVANTAGE),
        -c["emotion_macro_f1"],
    ))

    result = {
        "dataset": "CREMA-D DFA subset/cohort supplied to runner",
        "clips": len(files),
        "actors": int(len(np.unique(actor))),
        "model": "TAPF-MIN v2.2 chance-centered feature-level validation",
        "scientific_scope": "Controlled emotion-utility benchmark on 147-D motion input; not raw-video end-to-end clinical validation and not formal anonymity.",
        "protocol": {
            "utility": f"{folds}-fold actor-disjoint out-of-fold emotion prediction",
            "clip_privacy": "closed-set identity inference using Logistic, RBF-SVM, RandomForest, and independent MLP; below-chance AUC is reflected because attacker scores can be inverted",
            "privacy_metric": "identity_advantage = |AUC-0.50|; effective_auc = 0.50 + identity_advantage",
            "repeated_release_privacy": "two disjoint per-actor aggregates attacked after averaging repeated releases",
            "release_gate": "clip identity 95% CI entirely within AUC 0.45-0.55 AND repeated-release effective AUC <= 0.55 AND lower emotion F1 CI >= 0.20",
        },
        "targets": {
            "chance_identity_auc": CHANCE_AUC,
            "max_identity_advantage": MAX_IDENTITY_ADVANTAGE,
            "max_effective_auc": MAX_EFFECTIVE_AUC,
            "equivalent_auc_band": [0.45, 0.55],
            "utility_lower_f1": MIN_UTILITY_LOWER_F1,
        },
        "sweep": rows,
        "eligible_operating_points": eligible,
        "best_privacy_point": best_privacy,
        "best_utility_point": best_utility,
        "best_joint_point": best_joint,
        "edge_checks": {
            "actor_disjoint_utility": all(fr["actor_overlap"] == 0 for r in rows for fr in r["folds"]),
            "four_independent_posthoc_attackers": True,
            "below_chance_auc_reflected": True,
            "repeated_release_attack": True,
            "fail_closed": True,
            "raw_face_released": False,
        },
    }
    Path("results").mkdir(exist_ok=True)
    out = Path("results/cremad_final_validation.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    run(args.root, args.folds)
