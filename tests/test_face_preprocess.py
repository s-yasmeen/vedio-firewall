import numpy as np
from tapf.face_preprocess import FacePreprocessor


def test_missing_first_face_uses_neutral_not_scene_crop():
    p=FacePreprocessor(output_size=64)
    blank=np.zeros((100,140,3),dtype=np.uint8)
    r=p.crop_face(blank)
    assert r.crop.shape==(64,64,3)
    assert r.detected is False
    assert r.valid_face is False
    assert r.reused_bbox is False
    assert r.bbox is None
    assert np.all(r.crop==128)


def test_sequence_preserves_length_and_reports_quality_masks():
    p=FacePreprocessor(output_size=48)
    frames=[np.zeros((80,120,3),dtype=np.uint8) for _ in range(4)]
    crops,det,ali,valid,reused=p.process_sequence(frames)
    assert crops.shape==(4,48,48,3)
    for arr in (det,ali,valid,reused):
        assert arr.shape==(4,)
    assert not valid.any()
