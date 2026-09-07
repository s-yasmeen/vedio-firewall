"""Non-image minimum-disclosure representations for TAPF-MIN.

The goal is to avoid releasing a human-viewable face when a downstream task can operate
on compact temporal information. These features are research representations; privacy must
still be measured with an attacker trained on the released representation.
"""
from __future__ import annotations

import cv2
import numpy as np


def _gray_small(frame: np.ndarray, size=(48, 48)) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


def motion_descriptor(frames: list[np.ndarray], grid=(6, 6)) -> np.ndarray:
    """Return a compact, non-image temporal motion descriptor.

    Dense optical flow is pooled spatially and summarized across time. Absolute appearance
    is not released. The output contains per-grid-cell mean/std horizontal and vertical flow
    plus global temporal magnitude statistics.
    """
    if len(frames) < 2:
        raise ValueError("motion_descriptor requires at least two frames")
    prev = _gray_small(frames[0])
    pooled = []
    gh, gw = grid
    for frame in frames[1:]:
        cur = _gray_small(frame)
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 2, 9, 2, 5, 1.1, 0)
        h, w = flow.shape[:2]
        cells = []
        for iy in range(gh):
            y0, y1 = iy*h//gh, (iy+1)*h//gh
            for ix in range(gw):
                x0, x1 = ix*w//gw, (ix+1)*w//gw
                cell = flow[y0:y1, x0:x1]
                cells.extend([float(cell[...,0].mean()), float(cell[...,1].mean())])
        pooled.append(cells)
        prev = cur
    X = np.asarray(pooled, dtype=np.float32)
    mag = np.linalg.norm(X.reshape(len(X), -1, 2), axis=2)
    out = np.concatenate([X.mean(0), X.std(0), [mag.mean(), mag.std(), np.quantile(mag, .95)]])
    n = np.linalg.norm(out)
    return (out / n if n else out).astype(np.float32)


def read_motion_clip(path: str, every: int = 4, max_frames: int = 32) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames=[]; i=0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % max(1, every) == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"Not enough readable frames: {path}")
    return motion_descriptor(frames)
