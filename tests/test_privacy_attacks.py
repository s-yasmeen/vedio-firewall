import numpy as np
from tapf.privacy_attacks import verification_attack, linkage_attack

def test_identity_signal_is_detected():
    rng=np.random.default_rng(4); actors=np.repeat(np.arange(6),4)
    centers=np.eye(6); z=np.vstack([centers[a]+rng.normal(0,.02,6) for a in actors])
    assert verification_attack(z,actors)['effective_auc']>.9
    assert linkage_attack(z,actors)['rank1_accuracy']>.9

def test_attacks_report_bounded_metrics():
    rng=np.random.default_rng(8); actors=np.repeat(np.arange(5),4); z=rng.normal(size=(20,6))
    v=verification_attack(z,actors); l=linkage_attack(z,actors)
    assert .5<=v['effective_auc']<=1 and 0<=l['rank1_accuracy']<=1
