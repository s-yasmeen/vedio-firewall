"""Independent external validation of TAPF-MIN v2.2 on RAVDESS facial tracking.

Dataset: RAVDESS Facial Landmark Tracking (Zenodo 3255102). The benchmark uses the
speech trials from the official OpenFace tracking release. It does not require raw face
pixels: each trial is summarized from action-unit intensity, head-pose, and tracker
confidence time series.

Purpose:
- replicate the minimum-disclosure privacy/utility protocol on a dataset independent of
  CREMA-D,
- use actor-disjoint out-of-fold emotion utility,
- attack the released task representation for single-release and repeated-release identity,
- apply the same frozen TAPF-MIN v2.2 chance-centered gate.

This is external affective/biometric validation, not clinical validation.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.run_cremad_final_validation import _attack, repeated_release_attack
from experiments.run_cremad_privacy_utility_spectrum import bootstrap_metric
from tapf.privacy_gate_v22 import effective_auc, evaluate_v22_gate

# RAVDESS emotion codes.
EMOTIONS = np.asarray(["NEU", "CAL", "HAP", "SAD", "ANG", "FEA", "DIS", "SUR"])
NAME_RE = re.compile(r"^(01)-(01)-(0[1-8])-(0[12])-(0[12])-(0[12])-(\d{2})\.csv$", re.I)


def _selected_columns(header: list[str]) -> list[int]:
    h = [x.strip() for x in header]
    idx = []
    for i, name in enumerate(h):
        # OpenFace AU intensity channels, not binary presence channels.
        if re.match(r"^AU\d+_r$", name):
            idx.append(i)
        elif name in {"pose_Rx", "pose_Ry", "pose_Rz"}:
            idx.append(i)
    if not idx:
        raise RuntimeError("No action-unit/head-pose columns found in RAVDESS tracking CSV")
    return idx


def summarize_csv(raw: bytes) -> np.ndarray:
    text = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", errors="replace", newline="")
    reader = csv.reader(text)
    header = next(reader)
    stripped = [x.strip() for x in header]
    selected = _selected_columns(header)
    try:
        confidence_i = stripped.index("confidence")
    except ValueError:
        confidence_i = None

    rows = []
    confidences = []
    for row in reader:
        if not row or len(row) <= max(selected):
            continue
        try:
            vals = [float(row[i]) for i in selected]
            if not np.isfinite(vals).all():
                continue
            rows.append(vals)
            if confidence_i is not None and confidence_i < len(row):
                c = float(row[confidence_i])
                if np.isfinite(c):
                    confidences.append(c)
        except (ValueError, IndexError):
            continue

    X = np.asarray(rows, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 3:
        raise RuntimeError("Insufficient valid tracking frames")
    d = np.diff(X, axis=0)
    # Translation-invariant temporal summaries: levels and dynamics.
    feat = np.concatenate([
        X.mean(axis=0), X.std(axis=0),
        np.mean(np.abs(d), axis=0), d.std(axis=0),
        np.asarray([np.mean(confidences) if confidences else 0.0], dtype=np.float32),
    ])
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feat


def load_zip(path: str | Path, max_trials: int | None = None):
    vectors, actor, emotion, names = [], [], [], []
    with zipfile.ZipFile(path) as zf:
        members = []
        for name in zf.namelist():
            base = PurePosixPath(name).name
            m = NAME_RE.match(base)
            if not m:
                continue
            # Full audiovisual speech trials only: modality=01, channel=01 by regex.
            members.append((name, base, m))
        members.sort(key=lambda x: x[1])
        if max_trials:
            members = members[: int(max_trials)]
        for i, (name, base, m) in enumerate(members, 1):
            try:
                feat = summarize_csv(zf.read(name))
            except RuntimeError:
                continue
            vectors.append(feat)
            emotion_code = int(m.group(3)) - 1
            emotion.append(EMOTIONS[emotion_code])
            actor.append(m.group(7))
            names.append(base)
            if i % 100 == 0:
                print(f"processed {i}/{len(members)} trials", flush=True)
    X = np.asarray(vectors, dtype=np.float32)
    y_actor = np.asarray(actor)
    y_emotion = np.asarray(emotion)
    if len(X) < 200 or len(np.unique(y_actor)) < 12:
        raise RuntimeError(f"Insufficient RAVDESS cohort: {len(X)} trials/{len(np.unique(y_actor))} actors")
    return X, y_actor, y_emotion, names


def crossfit_task(X, actor, emotion, folds: int, seed: int):
    labels = np.asarray(sorted(np.unique(emotion)))
    label_to_i = {x: i for i, x in enumerate(labels)}
    P = np.zeros((len(X), len(labels)), dtype=np.float32)
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(actor))))
    fold_rows = []
    for fold, (tr, te) in enumerate(splitter.split(X, emotion, groups=actor)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, random_state=seed + fold),
        )
        model.fit(X[tr], emotion[tr])
        p = model.predict_proba(X[te])
        for j, cls in enumerate(model.classes_):
            P[te, label_to_i[cls]] = p[:, j]
        pred = labels[np.argmax(P[te], axis=1)]
        fold_rows.append({
            "fold": fold,
            "actors_test": sorted(np.unique(actor[te]).tolist()),
            "actor_overlap": int(len(set(actor[tr]) & set(actor[te]))),
            "macro_f1": float(f1_score(emotion[te], pred, average="macro", zero_division=0)),
        })
    return P, labels, fold_rows


def quantize(P: np.ndarray, levels: int) -> np.ndarray:
    q = np.rint(np.clip(P, 0, 1) * (levels - 1)) / (levels - 1)
    s = q.sum(axis=1, keepdims=True)
    zero = s[:, 0] == 0
    if np.any(zero):
        q[zero] = 1.0 / q.shape[1]
        s = q.sum(axis=1, keepdims=True)
    return (q / s).astype(np.float32)


def hard_label(P: np.ndarray) -> np.ndarray:
    return np.eye(P.shape[1], dtype=np.float32)[np.argmax(P, axis=1)]


def temperature(P: np.ndarray, t: float) -> np.ndarray:
    eps = 1e-8
    logits = np.log(np.clip(P, eps, 1.0)) / float(t)
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits); out /= out.sum(axis=1, keepdims=True)
    return out.astype(np.float32)


def evaluate_release(Z, actor, emotion, labels, seed: int, name: str):
    pred = labels[np.argmax(Z, axis=1)]
    f1, lo, hi = bootstrap_metric(
        emotion, pred,
        lambda y, p: f1_score(y, p, average="macro", zero_division=0),
        seed=seed, n=400,
    )
    clip = _attack(Z, actor, seed + 500)
    repeated = repeated_release_attack(Z, actor, seed + 900)
    attacker_aucs = [float(r["auc"]) for r in clip["attackers"]]
    gate = evaluate_v22_gate(
        clip_auc_ci95=clip["identity_auc_ci95"],
        repeated_release_auc=repeated["identity_auc"],
        task_f1_ci95=[lo, hi],
        attacker_aucs=attacker_aucs,
    )
    return {
        "release": name,
        "dimension": int(Z.shape[1]),
        "emotion_macro_f1": float(f1),
        "emotion_macro_f1_ci95": [float(lo), float(hi)],
        "clip_identity": clip,
        "repeated_release_identity": repeated,
        "clip_effective_auc": float(effective_auc(clip["identity_auc"])),
        "repeated_effective_auc": float(effective_auc(repeated["identity_auc"])),
        "v22_gate": gate,
    }


def run(zip_path: str | Path, folds: int = 5, seed: int = 42, max_trials: int | None = None):
    X, actor, emotion, names = load_zip(zip_path, max_trials=max_trials)
    P, labels, fold_rows = crossfit_task(X, actor, emotion, folds, seed)
    candidates = [
        ("posterior", P),
        ("posterior-q8", quantize(P, 8)),
        ("posterior-q4", quantize(P, 4)),
        ("posterior-q2", quantize(P, 2)),
        ("task-label-onehot", hard_label(P)),
        ("posterior-temp-1.5", temperature(P, 1.5)),
        ("posterior-temp-2.0", temperature(P, 2.0)),
        ("posterior-temp-4.0", temperature(P, 4.0)),
    ]
    rows = [evaluate_release(Z, actor, emotion, labels, seed + 1000*i, name)
            for i, (name, Z) in enumerate(candidates)]
    eligible = [r for r in rows if r["v22_gate"]["release_eligible"]]

    def privacy_rank(r):
        obs = r["v22_gate"]["observed"]
        return (max(obs["clip_effective_auc_upper"], obs["repeated_release_effective_auc"]),
                -obs["task_f1_lower_ci"], -r["emotion_macro_f1"])

    result = {
        "dataset": "RAVDESS Facial Landmark Tracking - audiovisual speech trials",
        "source": "Zenodo 10.5281/zenodo.3255102",
        "trials": int(len(X)),
        "actors": int(len(np.unique(actor))),
        "emotions": labels.tolist(),
        "raw_pixels_used": False,
        "input_representation": "OpenFace action-unit intensity + head-pose temporal summaries",
        "scope": "Independent external affective/biometric validation; not clinical validation.",
        "protocol": {
            "utility": f"{folds}-fold actor-disjoint emotion classification",
            "privacy": "four independent post-hoc identity attackers with chance-centered AUC",
            "repeated_release": "identity attack on disjoint per-actor aggregated task releases",
            "release_gate": "TAPF-MIN v2.2 frozen gate: clip effective-AUC upper <=0.55, repeated effective AUC <=0.55, every attacker <=0.55, task Macro-F1 lower 95% CI >=0.20",
        },
        "folds": fold_rows,
        "releases": rows,
        "eligible_count": len(eligible),
        "best_overall": min(rows, key=privacy_rank),
        "best_eligible": min(eligible, key=privacy_rank) if eligible else None,
        "best_utility": max(rows, key=lambda r: (r["emotion_macro_f1_ci95"][0], r["emotion_macro_f1"])),
        "integrity": {
            "actor_disjoint_utility": all(r["actor_overlap"] == 0 for r in fold_rows),
            "threshold_relaxation_after_results": False,
            "repeated_release_attack": True,
            "fail_closed": True,
        },
    }
    Path("results").mkdir(exist_ok=True)
    out = Path("results/ravdess_external_validation_v22.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-trials", type=int, default=None)
    args = ap.parse_args()
    run(args.zip_path, args.folds, args.seed, args.max_trials)
