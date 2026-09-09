"""Face-focused preprocessing for TAPF-MIN FER v3.1.

Research reference implementation using OpenCV's bundled face/eye cascades as a
lightweight fallback. It performs face detection, optional eye-based in-plane alignment,
and stable cropping. Production deployments should substitute a stronger landmark model
while preserving the same explicit detection/alignment reporting contract.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class FaceCropResult:
    crop: np.ndarray
    detected: bool
    aligned: bool
    bbox: tuple[int,int,int,int] | None


class FacePreprocessor:
    def __init__(self, *, output_size: int = 112, margin: float = 0.20):
        if output_size < 32: raise ValueError('output_size too small')
        if not 0 <= margin <= 1: raise ValueError('margin must be in [0,1]')
        self.output_size = int(output_size)
        self.margin = float(margin)
        face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        eye_path = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        self.detector = cv2.CascadeClassifier(face_path)
        self.eye_detector = cv2.CascadeClassifier(eye_path)
        if self.detector.empty() or self.eye_detector.empty():
            raise RuntimeError('OpenCV face/eye cascades unavailable')

    def _expand(self, x, y, w, h, width, height):
        mx, my = int(round(w*self.margin)), int(round(h*self.margin))
        x0=max(0,x-mx); y0=max(0,y-my); x1=min(width,x+w+mx); y1=min(height,y+h+my)
        return x0,y0,x1-x0,y1-y0

    def _align_by_eyes(self, crop_rgb: np.ndarray) -> tuple[np.ndarray, bool]:
        gray=cv2.cvtColor(crop_rgb,cv2.COLOR_RGB2GRAY)
        eyes=self.eye_detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4,minSize=(10,10))
        if len(eyes) < 2:
            return crop_rgb, False
        # Prefer two largest detections in the upper two-thirds of the face.
        h,w=gray.shape[:2]
        cand=[e for e in eyes if e[1] + e[3]/2 < 0.68*h]
        if len(cand) < 2: cand=list(eyes)
        cand=sorted(cand,key=lambda e:int(e[2])*int(e[3]),reverse=True)[:4]
        centers=[(e[0]+e[2]/2.0,e[1]+e[3]/2.0) for e in cand]
        best=None
        for i in range(len(centers)):
            for j in range(i+1,len(centers)):
                a,b=centers[i],centers[j]
                dx=abs(a[0]-b[0]); dy=abs(a[1]-b[1])
                if dx < 0.20*w: continue
                score=dx - 0.5*dy
                if best is None or score>best[0]: best=(score,a,b)
        if best is None: return crop_rgb, False
        _,a,b=best
        left,right=(a,b) if a[0] < b[0] else (b,a)
        angle=math.degrees(math.atan2(right[1]-left[1], right[0]-left[0]))
        center=((left[0]+right[0])/2.0,(left[1]+right[1])/2.0)
        M=cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned=cv2.warpAffine(crop_rgb,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        return aligned, True

    def crop_face(self, rgb: np.ndarray, *, previous_bbox=None) -> FaceCropResult:
        if rgb.ndim != 3 or rgb.shape[2] != 3: raise ValueError('rgb must be HxWx3')
        h,w = rgb.shape[:2]
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        faces=self.detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(32,32))
        detected=len(faces)>0
        bbox=None
        if detected:
            x,y,bw,bh=max(faces,key=lambda b:int(b[2])*int(b[3]))
            bbox=self._expand(int(x),int(y),int(bw),int(bh),w,h)
        elif previous_bbox is not None:
            x,y,bw,bh=map(int,previous_bbox)
            bbox=(max(0,x),max(0,y),min(bw,w-max(0,x)),min(bh,h-max(0,y)))
        else:
            side=min(h,w); x=(w-side)//2; y=(h-side)//2; bbox=(x,y,side,side)
        x,y,bw,bh=bbox
        crop=rgb[y:y+bh,x:x+bw]
        if crop.size == 0: raise RuntimeError('empty face crop')
        crop,aligned=self._align_by_eyes(crop) if detected else (crop,False)
        crop=cv2.resize(crop,(self.output_size,self.output_size),interpolation=cv2.INTER_AREA)
        return FaceCropResult(crop=crop,detected=detected,aligned=aligned,bbox=bbox)

    def process_sequence(self, frames_rgb):
        crops=[]; detections=[]; alignments=[]; prev=None
        for frame in frames_rgb:
            res=self.crop_face(frame,previous_bbox=prev)
            crops.append(res.crop); detections.append(bool(res.detected)); alignments.append(bool(res.aligned)); prev=res.bbox
        return np.asarray(crops), np.asarray(detections,dtype=bool), np.asarray(alignments,dtype=bool)
