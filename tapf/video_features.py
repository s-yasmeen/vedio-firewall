"""Lightweight video descriptors for empirical CPU baselines.

These descriptors are intentionally simple and reproducible. They provide a CPU baseline
for emotion utility and identity leakage, not a substitute for ArcFace/FaceNet or a modern
video-emotion network in the final publication threat model.
"""
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import hog


def frame_hog(frame: np.ndarray, size=(96, 96)) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    ).astype(np.float32)


def clip_descriptor(path: str, frame_stride: int = 5, max_frames: int = 48) -> np.ndarray:
    """Mean + standard deviation of frame HOG vectors across a video clip."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    feats = []
    frame_idx = 0
    used = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, frame_stride) == 0:
            feats.append(frame_hog(frame))
            used += 1
            if used >= max_frames:
                break
        frame_idx += 1
    cap.release()
    if not feats:
        raise RuntimeError(f"No readable frames: {path}")
    X = np.asarray(feats, dtype=np.float32)
    return np.concatenate([X.mean(axis=0), X.std(axis=0)]).astype(np.float32)
