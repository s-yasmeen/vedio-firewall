"""Server-side TAPF-MIN v3 release authority.

Binds release decisions to frozen evidence and a persistent privacy scope. Clients cannot
supply cumulative spend or validation metrics.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from tapf.evidence_registry import EvidenceRegistry
from tapf.formal_privacy import FormalPrivacyBudget, randomized_response, gaussian_release
from tapf.persistent_privacy import SQLiteDPAccountant

@dataclass(frozen=True)
class ReleasePolicyV3:
    min_utility_lower: float
    max_identity_risk_upper: float
    max_repeated_risk_upper: float
    epsilon_budget: float
    delta_budget: float=0.0

class ReleaseAuthority:
    def __init__(self, *, classes, registry: EvidenceRegistry, ledger_path, policy: ReleasePolicyV3):
        self.classes=tuple(classes); self.registry=registry; self.ledger_path=ledger_path; self.policy=policy
        if len(self.classes)<2: raise ValueError('at least two classes required')

    def release(self, *, scope_id, release_id, evidence_id, task, local_posterior, rng=None):
        e=self.registry.get(evidence_id,task=task)
        checks={
            'utility': e.utility_lower >= self.policy.min_utility_lower,
            'identity': e.identity_risk_upper <= self.policy.max_identity_risk_upper,
            'repeated': e.repeated_risk_upper <= self.policy.max_repeated_risk_upper,
        }
        if not all(checks.values()):
            return {'decision':'BLOCK','reason':'frozen_evidence_gate','checks':checks,'evidence_digest':e.digest}
        p=np.asarray(local_posterior,dtype=float).reshape(-1)
        if p.size!=len(self.classes) or not np.isfinite(p).all() or np.any(p<0) or p.sum()<=0:
            raise ValueError('invalid local posterior')
        p=p/p.sum()
        accountant=SQLiteDPAccountant(self.ledger_path,scope_id,FormalPrivacyBudget(self.policy.epsilon_budget,self.policy.delta_budget))
        # Spend atomically before producing a release; a crash may consume budget without disclosure,
        # which is conservative. release_id prevents replay/double spend.
        accountant.spend_once(release_id,e.mechanism,e.epsilon,e.delta)
        gen=rng or np.random.default_rng()
        if e.mechanism=='randomized_response_label':
            label=self.classes[int(np.argmax(p))]
            out=randomized_response(label,self.classes,epsilon=e.epsilon,rng=gen,accountant=None)
            payload={'type':'categorical_label','value':out['label']}
        elif e.mechanism=='gaussian_quantized_posterior':
            out=gaussian_release(p,epsilon=e.epsilon,delta=e.delta,clip_norm=1.0,rng=gen,accountant=None)
            x=np.maximum(np.asarray(out['values'],float),0); x=x/x.sum() if x.sum()>0 else np.full(len(x),1/len(x))
            bits=int(e.quantization_bits or 8); levels=(1<<bits)-1
            x=np.round(x*levels)/levels; x=x/x.sum() if x.sum()>0 else np.full(len(x),1/len(x))
            payload={'type':'quantized_posterior','values':x.tolist(),'quantization_bits':bits}
        else:
            raise ValueError('unsupported frozen mechanism')
        return {'decision':'RELEASE','payload':payload,'evidence_id':e.evidence_id,'evidence_digest':e.digest,
                'model_sha256':e.model_sha256,'protocol_sha256':e.protocol_sha256,
                'formal_privacy':accountant.statement(),'raw_video_released':False,'latent_released':False}
