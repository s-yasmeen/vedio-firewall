"""Empirical CPU baseline on transformed CREMA-D clips.

Trains two subject-disjoint classifiers from clip descriptors:
1) emotion utility classifier (authorized task),
2) actor identity attacker (privacy leakage proxy).

Use this as an initial empirical baseline only. Final publication evaluation should add
ArcFace/FaceNet-family identity attackers and a stronger emotion model.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from tapf.video_features import clip_descriptor


def load_descriptors(manifest_csv: str, method: str):
    df = pd.read_csv(manifest_csv)
    df = df[df["method"] == method].copy()
    rows, feats = [], []
    for _, row in df.iterrows():
        try:
            feat = clip_descriptor(row["protected_path"])
        except Exception as exc:
            print(f"SKIP {row['protected_path']}: {exc}")
            continue
        rows.append(row)
        feats.append(feat)
    if not feats:
        raise RuntimeError(f"No usable clips for method={method}")
    return pd.DataFrame(rows).reset_index(drop=True), np.asarray(feats)


def fit_eval_classifier(X, y, split, seed=42):
    train = np.where(split == "train")[0]
    test = np.where(split == "test")[0]
    if len(train) == 0 or len(test) == 0:
        raise RuntimeError("Manifest must contain train and test samples")
    clf = make_pipeline(StandardScaler(), LinearSVC(dual="auto", random_state=seed))
    clf.fit(X[train], y[train])
    pred = clf.predict(X[test])
    return {
        "accuracy": float(accuracy_score(y[test], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
        "macro_f1": float(f1_score(y[test], pred, average="macro")),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }


def run(manifest_csv, methods=("original", "blur", "pixelation", "static_tapf")):
    output = []
    for method in methods:
        meta, X = load_descriptors(manifest_csv, method)
        split = meta["split"].astype(str).to_numpy()
        emotion = fit_eval_classifier(X, meta["emotion"].astype(str).to_numpy(), split)
        identity = fit_eval_classifier(X, meta["actor_id"].astype(str).to_numpy(), split)
        output.append({
            "method": method,
            "emotion_macro_f1": emotion["macro_f1"],
            "emotion_balanced_accuracy": emotion["balanced_accuracy"],
            "identity_accuracy": identity["accuracy"],
            "identity_balanced_accuracy": identity["balanced_accuracy"],
            "attacker": "video-HOG+LinearSVC baseline",
            "utility_model": "video-HOG+LinearSVC baseline",
            "n_train": emotion["n_train"],
            "n_test": emotion["n_test"],
        })

    out = Path("results")
    out.mkdir(exist_ok=True)
    path = out / "cremad_empirical_baseline.json"
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print("NOTE: empirical CPU baseline; stronger independent attackers/models are required for final claims.")
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="protected_manifest.csv from run_cremad_tapf.py")
    args = ap.parse_args()
    run(args.manifest)
