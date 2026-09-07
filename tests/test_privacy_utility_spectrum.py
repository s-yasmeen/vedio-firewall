import numpy as np

from experiments.run_cremad_privacy_utility_spectrum import (
    identity_split,
    macro_ovr_auc,
    quantize_probabilities,
)


def test_identity_split_is_clip_disjoint_and_covers_each_actor():
    actor=np.asarray(["a"]*6+["b"]*6+["c"]*6)
    tr,te=identity_split(actor,seed=7)
    assert len(set(tr).intersection(set(te)))==0
    for a in np.unique(actor):
        assert np.any(actor[tr]==a)
        assert np.any(actor[te]==a)


def test_quantized_probabilities_remain_normalized_and_finite():
    p=np.asarray([[.01,.09,.20,.20,.20,.30],[.9,.02,.02,.02,.02,.02]],dtype=np.float32)
    for levels in (4,16):
        q=quantize_probabilities(p,levels)
        assert q.shape==p.shape
        assert np.isfinite(q).all()
        assert np.allclose(q.sum(1),1.0,atol=1e-6)
        assert np.all(q>=0)
        assert np.all(q<=1)


def test_macro_ovr_auc_is_one_for_perfect_multiclass_predictions():
    y=np.asarray(["a","b","c","a","b","c"])
    classes=np.asarray(["a","b","c"])
    p=np.zeros((len(y),3),dtype=float)
    for i,label in enumerate(y):
        p[i,np.flatnonzero(classes==label)[0]]=1.0
    assert macro_ovr_auc(y,p,classes)==1.0


def test_uniform_probabilities_are_chance_level():
    y=np.asarray(["a","b","c","a","b","c"])
    classes=np.asarray(["a","b","c"])
    p=np.full((len(y),3),1/3,dtype=float)
    assert macro_ovr_auc(y,p,classes)==0.5
