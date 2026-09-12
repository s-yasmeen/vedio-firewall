import numpy as np
import torch

from experiments.run_cremad_v24_composition_aware import (
    EMOTIONS,
    actor_output_matching,
    deterministic_task_smoothing,
    train_fold,
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


def test_train_fold_executes_end_to_end_on_synthetic_data():
    """Exercise the neural/adversarial training path, not only helper functions."""
    rng=np.random.default_rng(123)
    n_actors=8
    clips_per_actor=12
    actor=np.repeat(np.asarray([f"A{i:02d}" for i in range(n_actors)]), clips_per_actor)
    emotion=np.tile(np.asarray(list(EMOTIONS)*2), n_actors)
    X=rng.normal(size=(len(actor), 24)).astype(np.float32)
    # Inject task signal only so the smoke test is deterministic but does not depend on identity.
    emo_to_i={e:i for i,e in enumerate(EMOTIONS)}
    for i,e in enumerate(emotion):
        X[i, emo_to_i[e]] += 2.0
    train_mask=np.arange(len(actor)) < (n_actors-2)*clips_per_actor
    P=train_fold(
        X[train_mask], emotion[train_mask], actor[train_mask], X[~train_mask],
        seed=7, id_weight=0.75, center_weight=0.50, output_weight=2.0,
        aggregate_weight=0.75, noise_std=0.06, epochs=2,
    )
    assert P.shape == ((~train_mask).sum(), len(EMOTIONS))
    assert np.isfinite(P).all()
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-5)
