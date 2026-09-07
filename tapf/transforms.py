"""Image-space transforms for baseline and TAPF-MIN experiments.

Benchmark transforms retain a deterministic center fallback for reproducibility. Deployment
code must call ``apply_transform_strict`` so localization failure cannot silently pass as
verified biometric protection.
"""
from __future__ import annotations
import cv2
import numpy as np


class FaceLocalizationError(RuntimeError):
    pass


def _center_roi(frame: np.ndarray):
    h, w = frame.shape[:2]
    side = max(2, int(min(h, w) * 0.60))
    x = max(0, (w - side) // 2)
    y = max(0, (h - side) // 2)
    return x, y, side, side


def _detect_face_roi(frame: np.ndarray):
    """Return ``(roi, detected)`` without inventing a successful detection."""
    try:
        if not hasattr(cv2, "CascadeClassifier"):
            return None, False
        data = getattr(cv2, "data", None)
        haar_root = getattr(data, "haarcascades", None) if data is not None else None
        if not haar_root:
            return None, False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(haar_root + "haarcascade_frontalface_default.xml")
        if hasattr(detector, "empty") and detector.empty():
            return None, False
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return (int(x), int(y), int(w), int(h)), True
    except Exception:
        return None, False
    return None, False


def _face_roi(frame: np.ndarray):
    """Benchmark ROI: detector first, deterministic center fallback second."""
    roi, detected = _detect_face_roi(frame)
    return roi if detected else _center_roi(frame)


def _strict_face_roi(frame: np.ndarray):
    roi, detected = _detect_face_roi(frame)
    if not detected or roi is None:
        raise FaceLocalizationError("Face localization unavailable; deployment path must fail closed")
    return roi


def original(frame: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    return frame.copy()


def _blur_with_roi(frame: np.ndarray, alpha: float, roi) -> np.ndarray:
    out = frame.copy(); x, y, w, h = roi
    patch = out[y:y+h, x:x+w]
    k = max(3, int(7 + 40 * alpha)); k += 1 if k % 2 == 0 else 0
    out[y:y+h, x:x+w] = cv2.GaussianBlur(patch, (k, k), 0)
    return out


def blur(frame: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return _blur_with_roi(frame, alpha, _face_roi(frame))


def _pixelate_with_roi(frame: np.ndarray, alpha: float, roi) -> np.ndarray:
    out = frame.copy(); x, y, w, h = roi
    patch = out[y:y+h, x:x+w]
    scale = max(0.04, 0.35 * (1.0 - alpha) + 0.04)
    sw, sh = max(2, int(w * scale)), max(2, int(h * scale))
    small = cv2.resize(patch, (sw, sh), interpolation=cv2.INTER_LINEAR)
    out[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def pixelate(frame: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return _pixelate_with_roi(frame, alpha, _face_roi(frame))


def _static_tapf_with_roi(frame: np.ndarray, alpha: float, upper_ratio: float, roi) -> np.ndarray:
    out = frame.copy(); x, y, w, h = roi
    split = max(1, int(h * upper_ratio))
    upper = out[y:y+split, x:x+w]
    lower = out[y+split:y+h, x:x+w]
    k = max(3, int(9 + 36 * alpha)); k += 1 if k % 2 == 0 else 0
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


def static_tapf(frame: np.ndarray, alpha: float = 0.65, upper_ratio: float = 0.68) -> np.ndarray:
    """Experimental task-aware baseline; lower face is not identity-free."""
    return _static_tapf_with_roi(frame, alpha, upper_ratio, _face_roi(frame))


TRANSFORMS = {"original": original, "blur": blur, "pixelation": pixelate, "static_tapf": static_tapf}


def apply_transform(name: str, frame: np.ndarray, alpha: float) -> np.ndarray:
    if name not in TRANSFORMS:
        raise KeyError(f"Unknown transform: {name}")
    return TRANSFORMS[name](frame, alpha)


def apply_transform_strict(name: str, frame: np.ndarray, alpha: float) -> np.ndarray:
    """Release-critical transform path: any localization failure raises and must BLOCK."""
    if name == "original":
        raise FaceLocalizationError("Original/raw face is not a protected release transform")
    roi = _strict_face_roi(frame)
    if name == "blur":
        return _blur_with_roi(frame, alpha, roi)
    if name == "pixelation":
        return _pixelate_with_roi(frame, alpha, roi)
    if name == "static_tapf":
        return _static_tapf_with_roi(frame, alpha, 0.68, roi)
    raise KeyError(f"Unknown transform: {name}")
