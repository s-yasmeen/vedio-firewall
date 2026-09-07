"""Minimum-disclosure non-image representations for TAPF-MIN.

The functions in this module deliberately output numeric task features rather than
human-viewable face frames. They are research representations: privacy must still
be measured with an attacker trained on the released features.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def _gray_small(frame: np.ndarray, size=(32, 32)) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


def motion_task_representation(
    path: str | Path,
    frame_stride: int = 2,
    max_frames: int = 64,
    grid: int = 4,
) -> np.ndarray:
    """Return a compact temporal-change descriptor, not an image.

    Each sampled frame is resized to 32x32 grayscale. Absolute temporal changes
    are pooled into a grid. We summarize the sequence with mean/std/max change.
    This intentionally removes absolute appearance and most spatial detail while
    retaining coarse motion magnitude and distribution.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    previous = None
    changes: list[np.ndarray] = []
    idx = used = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(1, frame_stride) == 0:
            current = _gray_small(frame)
            if previous is not None:
                delta = np.abs(current - previous)
                h, w = delta.shape
                pooled = []
                for gy in range(grid):
                    for gx in range(grid):
                        cell = delta[gy*h//grid:(gy+1)*h//grid,
                                     gx*w//grid:(gx+1)*w//grid]
                        pooled.append(float(cell.mean()))
                changes.append(np.asarray(pooled, dtype=np.float32))
            previous = current
            used += 1
            if used >= max_frames:
                break
        idx += 1
    cap.release()

    if not changes:
        raise RuntimeError(f"Not enough readable frames: {path}")
    X = np.asarray(changes, dtype=np.float32)
    # Numeric representation only: 3 * grid^2 values (48 values at grid=4).
    return np.concatenate([X.mean(0), X.std(0), X.max(0)]).astype(np.float32)


def representation_metadata(grid: int = 4) -> dict:
    return {
        "representation": "motion-features",
        "human_viewable": False,
        "contains_raw_frames": False,
        "dimension": 3 * grid * grid,
        "privacy_claim": "empirical-only; must pass an independent identity attacker",
    }
