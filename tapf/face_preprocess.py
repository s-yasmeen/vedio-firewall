"""Face-focused preprocessing for TAPF-MIN FER v3.1.

Research reference implementation using OpenCV's bundled Haar cascade as a dependency-light
fallback. Production deployments should substitute a stronger detector/alignment model but
preserve the same fail-closed crop contract.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class FaceCropResult:
    crop: np.ndarray
    detected: bool
    bbox: tuple[int,int,int,int] | None


class FacePreprocessor:
    def __init__(self, *, output_size: int = 112, margin: float = 0.20):
        if output_size < 32: raise ValueError('output_size too small')
        if not 0 <= margin <= 1: raise ValueError('margin must be in [0,1]')
        self.output_size = int(output_size)
        self.margin = float(margin)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty(): raise RuntimeError('OpenCV face cascade unavailable')

    def _expand(self, x, y, w, h, width, height):
        mx, my = int(round(w*self.margin)), int(round(h*self.margin))
        x0=max(0,x-mx); y0=max(0,y-my); x1=min(width,x+w+mx); y1=min(height,y+h+my)
        return x0,y0,x1-x0,y1-y0

    def crop_face(self, rgb: np.ndarray, *, previous_bbox=None) -> FaceCropResult:
        if rgb.ndim != 3 or rgb.shape[2] != 3: raise ValueError('rgb must be HxWx3')
        h,w = rgb.shape[:2]
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        faces=self.detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(32,32))
        detected=len(faces)>0
        bbox=None
        if detected:
            # Largest face is most stable for single-participant telepresence-style video.
            x,y,bw,bh=max(faces,key=lambda b:int(b[2])*int(b[3]))
            bbox=self._expand(int(x),int(y),int(bw),int(bh),w,h)
        elif previous_bbox is not None:
            x,y,bw,bh=map(int,previous_bbox)
            bbox=(max(0,x),max(0,y),min(bw,w-max(0,x)),min(bh,h-max(0,y)))
        else:
            # Fail-soft research fallback: centered square, explicitly marked undetected.
            side=min(h,w); x=(w-side)//2; y=(h-side)//2; bbox=(x,y,side,side)
        x,y,bw,bh=bbox
        crop=rgb[y:y+bh,x:x+bw]
        if crop.size == 0: raise RuntimeError('empty face crop')
        crop=cv2.resize(crop,(self.output_size,self.output_size),interpolation=cv2.INTER_AREA)
        return FaceCropResult(crop=crop,detected=detected,bbox=bbox)

    def process_sequence(self, frames_rgb):
        crops=[]; detections=[]; prev=None
        for frame in frames_rgb:
            res=self.crop_face(frame,previous_bbox=prev)
            crops.append(res.crop); detections.append(bool(res.detected)); prev=res.bbox
        return np.asarray(crops), np.asarray(detections,dtype=bool)
