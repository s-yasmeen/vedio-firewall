import json, hashlib
import numpy as np
import pytest
from tapf.evidence_registry import FrozenEvidence, EvidenceRegistry
from tapf.release_authority import ReleaseAuthority, ReleasePolicyV3
from tapf.persistent_privacy import DuplicateRelease
from tapf.formal_privacy import PrivacyBudgetExceeded

H='a'*64; P='b'*64

def evidence(**kw):
    base=dict(evidence_id='fer-label-v1',model_sha256=H,protocol_sha256=P,task='emotion',mechanism='randomized_response_label',utility_lower=.80,identity_risk_upper=.04,repeated_risk_upper=.05,epsilon=.5)
    base.update(kw); return FrozenEvidence(**base)

def authority(tmp_path,e=None,budget=1.0):
    return ReleaseAuthority(classes=['A','B','C'],registry=EvidenceRegistry([e or evidence()]),ledger_path=tmp_path/'ledger.db',policy=ReleasePolicyV3(.70,.10,.10,budget,0.0))

def test_frozen_gate_blocks_without_spend(tmp_path):
    a=authority(tmp_path,evidence(identity_risk_upper=.3))
    r=a.release(scope_id='s',release_id='r1',evidence_id='fer-label-v1',task='emotion',local_posterior=[.8,.1,.1])
    assert r['decision']=='BLOCK'

def test_release_persists_budget_and_binds_hashes(tmp_path):
    a=authority(tmp_path)
    r=a.release(scope_id='patient-window-1',release_id='r1',evidence_id='fer-label-v1',task='emotion',local_posterior=[.8,.1,.1],rng=np.random.default_rng(1))
    assert r['decision']=='RELEASE' and r['model_sha256']==H
    b=authority(tmp_path)
    assert b.release(scope_id='patient-window-1',release_id='r2',evidence_id='fer-label-v1',task='emotion',local_posterior=[.8,.1,.1])['formal_privacy']['epsilon_spent']==1.0
    with pytest.raises(PrivacyBudgetExceeded):
        b.release(scope_id='patient-window-1',release_id='r3',evidence_id='fer-label-v1',task='emotion',local_posterior=[.8,.1,.1])

def test_replay_blocked(tmp_path):
    a=authority(tmp_path)
    a.release(scope_id='s',release_id='same',evidence_id='fer-label-v1',task='emotion',local_posterior=[1,0,0])
    with pytest.raises(DuplicateRelease):
        a.release(scope_id='s',release_id='same',evidence_id='fer-label-v1',task='emotion',local_posterior=[1,0,0])

def test_task_mismatch_fails(tmp_path):
    with pytest.raises(ValueError):
        authority(tmp_path).release(scope_id='s',release_id='r',evidence_id='fer-label-v1',task='other',local_posterior=[1,0,0])
