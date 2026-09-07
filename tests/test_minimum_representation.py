import numpy as np
import pytest

from tapf.minimum_representation import motion_descriptor


def _moving_square(dx=1, frames=6, size=64):
    out=[]
    for i in range(frames):
        frame=np.zeros((size,size,3),dtype=np.uint8)
        x=16+i*dx
        frame[20:36,x:x+16]=255
        out.append(frame)
    return out


def test_motion_descriptor_is_finite_and_fixed_size():
    v=motion_descriptor(_moving_square(),grid=(6,6))
    # mean/std for 2 flow components in 36 cells + 3 global magnitude stats
    assert v.shape==(147,)
    assert np.isfinite(v).all()
    assert np.linalg.norm(v)==pytest.approx(1.0,abs=1e-5)


def test_motion_descriptor_rejects_single_frame():
    with pytest.raises(ValueError):
        motion_descriptor([np.zeros((64,64,3),dtype=np.uint8)])


def test_motion_descriptor_changes_with_motion_direction():
    right=motion_descriptor(_moving_square(dx=1))
    left_frames=list(reversed(_moving_square(dx=1)))
    left=motion_descriptor(left_frames)
    assert not np.allclose(right,left,atol=1e-4)


def test_static_frames_do_not_create_nan():
    frame=np.zeros((64,64,3),dtype=np.uint8)
    v=motion_descriptor([frame.copy() for _ in range(4)])
    assert v.shape==(147,)
    assert np.isfinite(v).all()
    assert np.linalg.norm(v)==pytest.approx(0.0,abs=1e-7)
