import numpy as np

from tapf.privacy_amplification import privacy_blanket_sample, hard_task_release, ReleaseLedger


def test_hard_task_release_discards_confidence():
    P=np.array([[0.9,0.1],[0.2,0.8]],dtype=float)
    Z=hard_task_release(P)
    assert Z.tolist()==[[1.0,0.0],[0.0,1.0]]


def test_privacy_blanket_alpha_one_is_task_independent_in_distribution():
    P=np.tile(np.array([[0.999,0.001]],dtype=float),(5000,1))
    Z=privacy_blanket_sample(P,1.0,seed=7)
    rate=Z[:,0].mean()
    assert 0.46 < rate < 0.54


def test_privacy_blanket_reproducible_for_seed():
    P=np.array([[0.8,0.2],[0.3,0.7],[0.55,0.45]],dtype=float)
    a=privacy_blanket_sample(P,0.4,seed=11)
    b=privacy_blanket_sample(P,0.4,seed=11)
    assert np.array_equal(a,b)


def test_release_ledger_blocks_new_independent_evidence_after_budget():
    ledger=ReleaseLedger(max_new_releases_per_key=1)
    key=("session-1","emotion")
    assert ledger.allow_new_release(key) is True
    assert ledger.allow_new_release(key) is False
    assert ledger.status(key)["decision"]=="BLOCK_NEW_RELEASE"


def test_release_ledger_is_scoped_by_key():
    ledger=ReleaseLedger(max_new_releases_per_key=1)
    assert ledger.allow_new_release(("s1","emotion")) is True
    assert ledger.allow_new_release(("s2","emotion")) is True
