"""Representation-first TAPF-MIN edge pipeline.

Raw frames are processed locally. The pipeline emits task-specific derived representations
where possible and refuses to invent a representation when a required extractor/model is
missing. It intentionally does not provide network transport for raw frames.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable
import cv2
import numpy as np

from .deployment import allowed_representations
from .transforms import _strict_face_roi, FaceLocalizationError
from .cancelable import protect_embedding


@dataclass
class EdgeOutput:
    task: str
    representation: str
    payload: object
    raw_biometric_transmitted: bool = False


def _normalized_motion_features(frames: Iterable[np.ndarray]) -> np.ndarray:
    frames=list(frames)
    if len(frames)<2:
        raise ValueError("At least two frames are required for motion features")
    features=[]
    prev_gray=None
    for frame in frames:
        x,y,w,h=_strict_face_roi(frame)
        roi=frame[y:y+h,x:x+w]
        gray=cv2.resize(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY),(64,64))
        if prev_gray is not None:
            flow=cv2.calcOpticalFlowFarneback(prev_gray,gray,None,0.5,3,15,3,5,1.2,0)
            mag,ang=cv2.cartToPolar(flow[...,0],flow[...,1])
            features.append([
                float(np.mean(mag)), float(np.std(mag)),
                float(np.mean(np.cos(ang))), float(np.mean(np.sin(ang))),
            ])
        prev_gray=gray
    return np.asarray(features,dtype=np.float32)


def _rppg_proxy(frames: Iterable[np.ndarray]) -> np.ndarray:
    signal=[]
    for frame in frames:
        x,y,w,h=_strict_face_roi(frame)
        roi=frame[y:y+h,x:x+w]
        # Mean green-channel trace is a simple research proxy, not a validated clinical rPPG algorithm.
        signal.append(float(np.mean(roi[...,1])))
    arr=np.asarray(signal,dtype=np.float32)
    if arr.size:
        arr=arr-float(arr.mean())
        sd=float(arr.std())
        if sd>0: arr=arr/sd
    return arr


def derive_representation(task: str, frames: Iterable[np.ndarray], *,
                          action_unit_extractor: Callable | None=None,
                          embedding_extractor: Callable | None=None,
                          cancelable_secret: str | None=None) -> EdgeOutput:
    task_key=task.strip().lower(); frames=list(frames)
    allowed=allowed_representations(task_key)

    if task_key in {"movement","neurology-motion"}:
        return EdgeOutput(task_key,"motion-features",_normalized_motion_features(frames))

    if task_key=="rppg":
        return EdgeOutput(task_key,"physiological-signal",_rppg_proxy(frames))

    if task_key in {"facial-expression","emotion"}:
        if action_unit_extractor is None:
            raise RuntimeError("Action-unit extractor is required; fail closed rather than transmit a face")
        payload=action_unit_extractor(frames)
        return EdgeOutput(task_key,"action-units",payload)

    if task_key=="authentication":
        if embedding_extractor is None or not cancelable_secret:
            raise RuntimeError("Embedding extractor and cancelable secret are required for authentication")
        embedding=embedding_extractor(frames)
        protected=protect_embedding(embedding,cancelable_secret)
        return EdgeOutput(task_key,"cancelable-template",{
            "values":protected.values.tolist(),"key_id":protected.key_id
        })

    if task_key=="clinician-visual":
        raise RuntimeError("Clinician visual release requires a separately verified protected-video pipeline")

    raise RuntimeError(f"No minimum-disclosure representation implemented for task={task_key!r}; allowed={allowed}")
