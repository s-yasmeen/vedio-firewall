"""Synthetic clinical-like validation track for TAPF-MIN v2.2.

Purpose: controlled stress testing of privacy/utility logic before access to real clinical
data. This is NOT clinical validation and must never be reported as evidence of diagnostic
safety or efficacy.

The generator creates longitudinal patient sessions with separable latent components:
- patient identity nuisance,
- clinical-like movement severity signal,
- session/measurement noise,
- optional identity/task confounding.

We train the authorized task model on one patient cohort and evaluate on entirely unseen
patients. Identity attackers then receive only the released task representation and are
trained/tested on disjoint sessions of the held-out patients, including an aggregate
repeated-release attack.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tapf.privacy_gate_v22 import effective_auc, evaluate_v22_gate
from experiments.run_cremad_final_validation import _attack, repeated_release_attack
from experiments.run_cremad_privacy_utility_spectrum import bootstrap_metric


def make_synthetic_cohort(
    *,
    n_patients: int = 120,
    sessions_per_patient: int = 8,
    feature_dim: int = 64,
    n_classes: int = 4,
    identity_strength: float = 1.5,
    task_strength: float = 1.5,
    confounding: float = 0.0,
    noise: float = 1.0,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    patient_ids = np.arange(n_patients)
    severity = rng.integers(0, n_classes, size=n_patients)

    # Orthonormal-ish independent subspaces for identity and task signal.
    q, _ = np.linalg.qr(rng.normal(size=(feature_dim, feature_dim)))
    id_basis = q[:, : min(20, feature_dim // 2)]
    task_basis = q[:, min(20, feature_dim // 2): min(20, feature_dim // 2) + n_classes]
    id_latent = rng.normal(size=(n_patients, id_basis.shape[1]))

    task_codes = np.eye(n_classes)[severity]
    # Optional confounding makes a fraction of the clinical code identity-dependent.
    conf = rng.normal(size=(n_patients, n_classes))
    mixed_task = (1.0 - confounding) * task_codes + confounding * conf

    X, y, actor = [], [], []
    for p in patient_ids:
        identity_vec = identity_strength * (id_basis @ id_latent[p])
        clinical_vec = task_strength * (task_basis @ mixed_task[p])
        for s in range(sessions_per_patient):
            drift = 0.15 * np.sin((s + 1) / sessions_per_patient * 2 * np.pi)
            x = identity_vec + (1.0 + drift) * clinical_vec + rng.normal(scale=noise, size=feature_dim)
            X.append(x.astype(np.float32)); y.append(int(severity[p])); actor.append(f"P{p:04d}")
    return np.asarray(X), np.asarray(y), np.asarray(actor)


def patient_disjoint_split(actor: np.ndarray, seed: int = 42, train_fraction: float = 0.70):
    rng = np.random.default_rng(seed)
    patients = np.unique(actor).copy(); rng.shuffle(patients)
    cut = max(2, int(round(len(patients) * train_fraction)))
    tr_pat = set(patients[:cut]); te_pat = set(patients[cut:])
    tr = np.asarray([a in tr_pat for a in actor])
    te = np.asarray([a in te_pat for a in actor])
    return tr, te


def release_posterior(X, y, actor, seed: int = 42):
    tr, te = patient_disjoint_split(actor, seed)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
    )
    model.fit(X[tr], y[tr])
    P = model.predict_proba(X[te]).astype(np.float32)
    return P, y[te], actor[te]


def quantize(P: np.ndarray, levels: int) -> np.ndarray:
    q = np.round(np.asarray(P) * (levels - 1)) / (levels - 1)
    s = q.sum(axis=1, keepdims=True)
    zero = s[:, 0] <= 0
    if np.any(zero): q[zero] = 1.0 / q.shape[1]; s = q.sum(axis=1, keepdims=True)
    return (q / s).astype(np.float32)


def hard_label(P: np.ndarray) -> np.ndarray:
    idx = np.argmax(P, axis=1)
    return np.eye(P.shape[1], dtype=np.float32)[idx]


def randomized_label(P: np.ndarray, keep: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.argmax(P, axis=1).copy(); k = P.shape[1]
    for i in range(len(idx)):
        if rng.random() > keep:
            choices = np.arange(k); choices = choices[choices != idx[i]]
            idx[i] = int(rng.choice(choices))
    return np.eye(k, dtype=np.float32)[idx]


def evaluate_release(Z, y, actor, seed: int, label: str):
    pred = np.argmax(Z, axis=1)
    f1, flo, fhi = bootstrap_metric(
        y, pred, lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        seed=seed, n=400,
    )
    clip = _attack(Z, actor, seed + 100, bootstrap_n=300)
    repeated = repeated_release_attack(Z, actor, seed + 200)
    gate = evaluate_v22_gate(
        clip_auc_ci95=clip["identity_auc_ci95"],
        repeated_release_auc=repeated["identity_auc"],
        task_f1_ci95=[flo, fhi],
        attacker_aucs=[r["auc"] for r in clip["attackers"]],
    )
    return {
        "release": label,
        "task_macro_f1": float(f1),
        "task_macro_f1_ci95": [float(flo), float(fhi)],
        "clip_identity_auc": float(clip["identity_auc"]),
        "clip_identity_auc_ci95": [float(x) for x in clip["identity_auc_ci95"]],
        "clip_effective_auc": float(effective_auc(clip["identity_auc"])),
        "repeated_identity_auc": float(repeated["identity_auc"]),
        "repeated_effective_auc": float(effective_auc(repeated["identity_auc"])),
        "worst_clip_attacker": clip["worst_case_attacker"],
        "v22_gate": gate,
    }


def run(seed: int = 42):
    scenarios = []
    for confounding in (0.0, 0.20, 0.40):
        X, y, actor = make_synthetic_cohort(confounding=confounding, seed=seed + int(confounding * 1000))
        P, yt, at = release_posterior(X, y, actor, seed + 10)
        candidates = [
            ("posterior", P),
            ("posterior-q4", quantize(P, 4)),
            ("posterior-q2", quantize(P, 2)),
            ("task-label", hard_label(P)),
            ("task-label-randomized-p0.90", randomized_label(P, 0.90, seed + 101)),
            ("task-label-randomized-p0.80", randomized_label(P, 0.80, seed + 102)),
            ("task-label-randomized-p0.70", randomized_label(P, 0.70, seed + 103)),
        ]
        rows = [evaluate_release(Z, yt, at, seed + 500 + i * 31, name) for i, (name, Z) in enumerate(candidates)]
        eligible = [r for r in rows if r["v22_gate"]["release_eligible"]]
        scenarios.append({
            "identity_task_confounding": confounding,
            "held_out_patients": int(len(np.unique(at))),
            "held_out_sessions": int(len(at)),
            "releases": rows,
            "eligible_releases": [r["release"] for r in eligible],
        })

    result = {
        "benchmark": "TAPF-MIN v2.2 synthetic clinical-like stress validation",
        "claim_boundary": "Synthetic controlled validation only; NOT clinical validation, diagnostic validation, or patient-safety evidence.",
        "task": "four-level synthetic movement-severity classification",
        "privacy_target": "chance-centered effective identity AUC <= 0.55 including repeated-release attack and all tested attackers",
        "utility_target": "lower 95% CI of task Macro-F1 >= 0.20",
        "patient_disjoint_task_split": True,
        "scenarios": scenarios,
    }
    Path("results").mkdir(exist_ok=True)
    out = Path("results/synthetic_clinical_stress_v22.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); run(args.seed)
