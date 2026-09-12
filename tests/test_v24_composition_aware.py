import numpy as np
import torch

from experiments.run_cremad_v24_composition_aware import (
    actor_output_matching,
    deterministic_task_smoothing,
)


def test_session_stable_release_is_deterministic():
    P=np.asarray([[0.7,0.2,0.1],[0.1,0.8,0.1]],dtype=np.float32)
    a=deterministic_task_smoothing(P,0.25)
    b=deterministic_task_smoothing(P,0.25)
    assert np.array_equal(a,b)
    assert np.allclose(a.sum(axis=1),1.0)


def test_smoothing_preserves_predicted_task():
    P=np.asarray([[0.7,0.2,0.1],[0.1,0.8,0.1]],dtype=np.float32)
    for alpha in (0.0,0.1,0.4,0.55):
        Z=deterministic_task_smoothing(P,alpha)
        assert np.array_equal(np.argmax(Z,axis=1),np.argmax(P,axis=1))


def test_actor_output_matching_penalizes_actor_specific_signature():
    # Same true task, actor 0 predicts class 0 while actor 1 predicts class 1.
    logits=torch.tensor([[5.,0.],[5.,0.],[0.,5.],[0.,5.]])
    y_task=torch.tensor([0,0,0,0])
    y_actor=torch.tensor([0,0,1,1])
    bad=float(actor_output_matching(logits,y_task,y_actor))
    # Identical actor output distributions should have no actor-specific penalty.
    logits2=torch.tensor([[5.,0.],[5.,0.],[5.,0.],[5.,0.]])
    good=float(actor_output_matching(logits2,y_task,y_actor))
    assert bad > good
    assert good < 1e-8
