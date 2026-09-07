"""Real-data LFW verification benchmark for TAPF-MIN transforms.

LFW is already a cropped-face benchmark, so this script deliberately applies each
protection transform to the full crop. Privacy leakage is measured as pairwise
identity-verification ROC-AUC using normalized HOG descriptors. Lower AUC means
less identity separability under this lightweight attacker. EER is also reported.

This remains a baseline attacker, not a substitute for ArcFace/FaceNet evaluation.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import roc_auc_score, roc_curve
from skimage.feature import hog

METHODS = {
    "original": 0.0,
    "blur": 0.85,
    "pixelation": 0.85,
    "static_tapf": 0.65,
}


def to_bgr(gray):
    img = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def full_crop_transform(name, frame, alpha):
    out = frame.copy()
    h, w = out.shape[:2]
    if name == "original":
        return out
    if name == "blur":
        k = max(3, int(7 + 40 * alpha)); k += (k % 2 == 0)
        return cv2.GaussianBlur(out, (k, k), 0)
    if name == "pixelation":
        scale = max(0.04, 0.35 * (1.0 - alpha) + 0.04)
        sw, sh = max(2, int(w * scale)), max(2, int(h * scale))
        small = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if name == "static_tapf":
        split = max(1, int(h * 0.68))
        upper = out[:split].copy(); lower = out[split:].copy()
        k = max(3, int(9 + 36 * alpha)); k += (k % 2 == 0)
        upper = cv2.GaussianBlur(upper, (k, k), 0)
        if lower.size:
            local_alpha = max(0.05, alpha * 0.35)
            scale = max(0.10, 0.45 * (1.0 - local_alpha))
            sw, sh = max(2, int(w * scale)), max(2, int(lower.shape[0] * scale))
            low = cv2.resize(lower, (sw, sh), interpolation=cv2.INTER_LINEAR)
            lower = cv2.resize(low, (w, lower.shape[0]), interpolation=cv2.INTER_NEAREST)
        out[:split] = upper
        if lower.size: out[split:] = lower
        return out
    raise KeyError(name)


def descriptor(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 96))
    v = hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def make_pairs(y, max_genuine=2000, max_impostor=2000, seed=42):
    rng = np.random.default_rng(seed)
    by_id = {lab: np.flatnonzero(y == lab) for lab in np.unique(y)}
    genuine = []
    for idxs in by_id.values():
        if len(idxs) < 2: continue
        for _ in range(min(len(idxs) * 3, 60)):
            a, b = rng.choice(idxs, 2, replace=False)
            genuine.append((int(a), int(b), 1))
    rng.shuffle(genuine); genuine = genuine[:max_genuine]

    labels = list(by_id.keys())
    impostor = []
    while len(impostor) < min(max_impostor, max(1, len(genuine))):
        la, lb = rng.choice(labels, 2, replace=False)
        a = int(rng.choice(by_id[la])); b = int(rng.choice(by_id[lb]))
        impostor.append((a, b, 0))
    return genuine + impostor


def eer_from_scores(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def run(min_faces=20, resize=0.7, seed=42):
    data = fetch_lfw_people(min_faces_per_person=min_faces, resize=resize, color=False)
    frames = [to_bgr(img) for img in data.images]
    y = data.target
    pairs = make_pairs(y, seed=seed)
    labels = np.asarray([p[2] for p in pairs], dtype=int)

    results = []
    for method, alpha in METHODS.items():
        transformed = [full_crop_transform(method, f, alpha) for f in frames]
        X = np.asarray([descriptor(f) for f in transformed])
        scores = np.asarray([float(np.dot(X[a], X[b])) for a, b, _ in pairs])
        auc = float(roc_auc_score(labels, scores))
        eer = eer_from_scores(labels, scores)
        delta = float(np.mean([np.mean(np.abs(transformed[i].astype(np.float32)-frames[i].astype(np.float32))) for i in range(min(100, len(frames)))]))
        results.append({
            "method": method,
            "alpha": alpha,
            "n_samples": int(len(y)),
            "n_identities": int(len(np.unique(y))),
            "n_pairs": int(len(pairs)),
            "identity_verification_auc": auc,
            "identity_eer": eer,
            "mean_pixel_change_first100": delta,
            "attacker": "normalized HOG cosine verification baseline",
        })

    out = Path("results"); out.mkdir(exist_ok=True)
    (out / "lfw_privacy_baseline.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("NOTE: empirical lightweight verification benchmark; publication claims require modern independent attackers.")
    return results


if __name__ == "__main__":
    run()
