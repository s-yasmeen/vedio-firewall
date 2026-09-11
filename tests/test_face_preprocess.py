import numpy as np
from tapf.face_preprocess import FacePreprocessor, FaceCropResult


def test_missing_first_face_uses_neutral_not_scene_crop():
    p = FacePreprocessor(output_size=64)
    blank = np.zeros((100, 140, 3), dtype=np.uint8)
    r = p.crop_face(blank)
    assert r.crop.shape == (64, 64, 3)
    assert r.detected is False
    assert r.valid_face is False
    assert r.reused_bbox is False
    assert r.bbox is None
    assert np.all(r.crop == 128)


def test_sequence_preserves_length_and_reports_quality_masks():
    p = FacePreprocessor(output_size=48)
    frames = [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(4)]
    crops, det, ali, valid, reused = p.process_sequence(frames)
    assert crops.shape == (4, 48, 48, 3)
    for arr in (det, ali, valid, reused):
        assert arr.shape == (4,)
    assert not valid.any()


def test_reuse_streak_is_bounded():
    p = FacePreprocessor(output_size=32, max_reuse_frames=2)
    neutral = np.full((32, 32, 3), 128, dtype=np.uint8)
    calls = {'n': 0}

    def fake_crop(_frame, *, previous_bbox=None):
        calls['n'] += 1
        if calls['n'] == 1:
            return FaceCropResult(neutral, True, False, True, False, (1, 1, 20, 20))
        if previous_bbox is not None:
            return FaceCropResult(neutral, False, False, True, True, previous_bbox)
        return FaceCropResult(neutral, False, False, False, False, None)

    p.crop_face = fake_crop
    frames = [neutral] * 5
    _, det, _, valid, reused = p.process_sequence(frames)
    assert det.tolist() == [True, False, False, False, False]
    assert reused.tolist() == [False, True, True, False, False]
    assert valid.tolist() == [True, True, True, False, False]


def test_invalid_reuse_configuration_rejected():
    try:
        FacePreprocessor(output_size=32, max_reuse_frames=6)
        assert False, 'expected ValueError'
    except ValueError:
        pass
