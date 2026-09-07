import numpy as np
from tapf.transforms import apply_transform


def test_all_transforms_preserve_shape():
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    frame[35:125, 40:120] = 180
    for method, alpha in [("original",0.0),("blur",0.8),("pixelation",0.8),("static_tapf",0.65)]:
        out = apply_transform(method, frame, alpha)
        assert out.shape == frame.shape
        assert out.dtype == frame.dtype
