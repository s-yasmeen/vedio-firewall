import numpy as np
from tapf.face_preprocess import FacePreprocessor


def test_center_fallback_is_explicitly_undetected():
    p=FacePreprocessor(output_size=64)
    blank=np.zeros((100,140,3),dtype=np.uint8)
    r=p.crop_face(blank)
    assert r.crop.shape==(64,64,3)
    assert r.detected is False
    assert r.bbox is not None


def test_sequence_preserves_length():
    p=FacePreprocessor(output_size=48)
    frames=[np.zeros((80,120,3),dtype=np.uint8) for _ in range(4)]
    crops,det=p.process_sequence(frames)
    assert crops.shape==(4,48,48,3)
    assert det.shape==(4,)
