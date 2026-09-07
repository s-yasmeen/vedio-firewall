"""Real-data LFW privacy benchmark for TAPF-MIN transforms.

This script downloads LFW through scikit-learn, extracts HOG descriptors, and measures
identity classification accuracy under Original/Blur/Pixelation/Static TAPF.

The HOG classifier is a lightweight baseline attacker, NOT the final ArcFace/FaceNet threat model.
Its results are empirical for this protocol and must be labelled accordingly.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from skimage.feature import hog

from tapf.transforms import apply_transform

METHODS = {
    "original": 0.0,
    "blur": 0.85,
    "pixelation": 0.85,
    "static_tapf": 0.65,
}


def to_bgr(gray):
    img = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def descriptor(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 96))
    return hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)


def run(min_faces=20, resize=0.7, test_size=0.30, seed=42):
    data = fetch_lfw_people(min_faces_per_person=min_faces, resize=resize, color=False)
    frames = [to_bgr(img) for img in data.images]
    y = data.target
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)

    results = []
    for method, alpha in METHODS.items():
        transformed = [apply_transform(method, f, alpha) for f in frames]
        X = np.asarray([descriptor(f) for f in transformed])
        clf = make_pipeline(StandardScaler(), LinearSVC(dual="auto", random_state=seed))
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        results.append({
            "method": method,
            "alpha": alpha,
            "n_samples": int(len(y)),
            "n_identities": int(len(np.unique(y))),
            "identity_accuracy": float(accuracy_score(y[test_idx], pred)),
            "balanced_identity_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
            "attacker": "HOG+LinearSVC baseline",
        })

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "lfw_privacy_baseline.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("NOTE: empirical LFW baseline attacker; replace/add ArcFace and FaceNet for publication claims.")
    return results


if __name__ == "__main__":
    run()
