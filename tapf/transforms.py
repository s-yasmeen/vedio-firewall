"""Image-space transforms for baseline and TAPF-MIN experiments."""
from __future__ import annotations
import cv2
import numpy as np


def _face_roi(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return int(x), int(y), int(w), int(h)
    h, w = frame.shape[:2]
    side = int(min(h, w) * 0.60)
    x = max(0, (w - side) // 2)
    y = max(0, (h - side) // 2)
    return x, y, side, side


def original(frame: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    return frame.copy()


def blur(frame: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    out = frame.copy()
    x, y, w, h = _face_roi(out)
    roi = out[y:y+h, x:x+w]
    k = max(3, int(7 + 40 * alpha))
    if k % 2 == 0:
        k += 1
    out[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def pixelate(frame: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    out = frame.copy()
    x, y, w, h = _face_roi(out)
    roi = out[y:y+h, x:x+w]
    scale = max(0.04, 0.35 * (1.0 - alpha) + 0.04)
    sw, sh = max(2, int(w * scale)), max(2, int(h * scale))
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
    out[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def static_tapf(frame: np.ndarray, alpha: float = 0.65, upper_ratio: float = 0.68) -> np.ndarray:
    """Prototype task-aware transform: stronger protection in upper face.

    This is an experimental transform policy; it is not a claim that the lower face is identity-free.
    """
    out = frame.copy()
    x, y, w, h = _face_roi(out)
    split = max(1, int(h * upper_ratio))
    upper = out[y:y+split, x:x+w]
    lower = out[y+split:y+h, x:x+w]

    k = max(3, int(9 + 36 * alpha))
    if k % 2 == 0:
        k += 1
    upper = cv2.GaussianBlur(upper, (k, k), 0)

    if lower.size:
        local_alpha = max(0.05, alpha * 0.35)
        scale = max(0.10, 0.45 * (1.0 - local_alpha))
        sw, sh = max(2, int(w * scale)), max(2, int(lower.shape[0] * scale))
        low = cv2.resize(lower, (sw, sh), interpolation=cv2.INTER_LINEAR)
        lower = cv2.resize(low, (w, lower.shape[0]), interpolation=cv2.INTER_NEAREST)

    out[y:y+split, x:x+w] = upper
    if lower.size:
        out[y+split:y+h, x:x+w] = lower
    return out


TRANSFORMS = {
    "original": original,
    "blur": blur,
    "pixelation": pixelate,
    "static_tapf": static_tapf,
}


def apply_transform(name: str, frame: np.ndarray, alpha: float) -> np.ndarray:
    if name not in TRANSFORMS:
        raise KeyError(f"Unknown transform: {name}")
    return TRANSFORMS[name](frame, alpha)
