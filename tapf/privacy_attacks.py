"""Additional empirical attacks on released task representations.
These are empirical stress tests, not formal privacy guarantees.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score

def _norm(x):
    x=np.asarray(x,float); n=np.linalg.norm(x,axis=1,keepdims=True); return x/np.maximum(n,1e-12)

def verification_attack(z, actor):
    """Pairwise same-person verification AUC using cosine similarity."""
    z=_norm(z); actor=np.asarray(actor); scores=[]; labels=[]
    for i in range(len(z)):
        for j in range(i+1,len(z)):
            scores.append(float(z[i]@z[j])); labels.append(int(actor[i]==actor[j]))
    if len(set(labels))<2: raise ValueError('need same and different identity pairs')
    auc=float(roc_auc_score(labels,scores)); eff=max(auc,1-auc)
    return {'auc':auc,'effective_auc':eff,'identity_advantage':eff-.5,'pairs':len(labels)}

def linkage_attack(z, actor, seed=42):
    """Closed-world cross-session linkage: nearest enrollment centroid, one half vs the other."""
    z=_norm(z); actor=np.asarray(actor); rng=np.random.default_rng(seed); centroids={}; probes=[]; truth=[]
    for a in sorted(np.unique(actor)):
        ids=np.flatnonzero(actor==a).copy(); rng.shuffle(ids)
        if len(ids)<2: continue
        cut=max(1,len(ids)//2); centroids[a]=_norm(np.mean(z[ids[:cut]],axis=0,keepdims=True))[0]
        for i in ids[cut:]: probes.append(z[i]); truth.append(a)
    names=np.asarray(list(centroids)); C=np.asarray([centroids[a] for a in names]); probes=np.asarray(probes)
    pred=names[np.argmax(probes@C.T,axis=1)]
    acc=float(np.mean(pred==np.asarray(truth)))
    chance=1.0/max(1,len(names))
    return {'rank1_accuracy':acc,'chance_rank1':chance,'identity_advantage_over_chance':max(0.0,acc-chance),'identities':len(names),'probes':len(probes)}
