"""Evaluate a non-image TAPF-MIN motion representation on CREMA-D.

The released object is a compact motion feature vector, not face pixels. Privacy is tested
with an adaptive attacker trained directly on that released representation. Utility is
estimated with actor-disjoint cross-validation. No anonymity or clinical-validity claim is
made; a representation is eligible only after predeclared privacy/utility criteria are met.
"""
from pathlib import Path
import argparse, json, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tapf.minimum_representation import read_motion_clip

NAME_RE=re.compile(r"^(\d{4})_DFA_(ANG|DIS|FEA|HAP|NEU|SAD)_XX\.flv$",re.I)
MIN_REAL_VIDEO_BYTES=1000


def bootstrap_f1(y, pred, seed=42, n=500):
    rng=np.random.default_rng(seed); y=np.asarray(y); pred=np.asarray(pred); vals=[]
    point=float(f1_score(y,pred,average="macro",zero_division=0))
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y))
        vals.append(float(f1_score(y[idx],pred[idx],average="macro",zero_division=0)))
    lo,hi=np.quantile(vals,[.025,.975])
    return point,float(lo),float(hi)


def macro_ovr_auc(y_true, proba, classes):
    vals=[]
    for j,c in enumerate(classes):
        yy=(np.asarray(y_true)==c).astype(int)
        if yy.min()==yy.max():
            continue
        vals.append(roc_auc_score(yy,proba[:,j]))
    if not vals:
        raise ValueError("No valid one-vs-rest class AUCs")
    return float(np.mean(vals))


def bootstrap_auc(y, proba, classes, seed=43, n=500):
    rng=np.random.default_rng(seed); y=np.asarray(y); proba=np.asarray(proba); vals=[]
    point=macro_ovr_auc(y,proba,classes)
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y))
        try: vals.append(macro_ovr_auc(y[idx],proba[idx],classes))
        except ValueError: pass
    lo,hi=np.quantile(vals,[.025,.975]) if vals else (point,point)
    return point,float(lo),float(hi)


def run(root, folds=5, seed=42):
    root=Path(root); files=[]
    for p in sorted(root.glob("*.flv")):
        m=NAME_RE.match(p.name)
        if m and p.stat().st_size>MIN_REAL_VIDEO_BYTES:
            files.append((p,m.group(1),m.group(2)))
    actors=sorted({a for _,a,_ in files})
    if len(files)<120 or len(actors)<8:
        raise RuntimeError(f"Need >=120 hydrated clips and >=8 actors; got {len(files)} clips/{len(actors)} actors")
    folds=max(2,min(int(folds),len(actors)))

    X=np.asarray([read_motion_clip(str(p)) for p,_,_ in files],dtype=np.float32)
    actor=np.asarray([a for _,a,_ in files]); emotion=np.asarray([e for _,_,e in files])
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite released features detected")

    # Utility: every test fold contains actors never used to train that fold's task model.
    oof=np.empty(len(files),dtype=emotion.dtype)
    splitter=GroupKFold(n_splits=folds)
    fold_rows=[]
    for fold,(tr,te) in enumerate(splitter.split(X,emotion,groups=actor),1):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced",random_state=seed+fold))
        clf.fit(X[tr],emotion[tr]); oof[te]=clf.predict(X[te])
        fold_rows.append({"fold":fold,"train_actors":int(len(np.unique(actor[tr]))),"test_actors":int(len(np.unique(actor[te]))),
                          "macro_f1":float(f1_score(emotion[te],oof[te],average="macro",zero_division=0))})
    f1,f1_lo,f1_hi=bootstrap_f1(emotion,oof,seed+1)

    # Adaptive closed-set attacker: for each identity, train/test clips are disjoint.
    rng=np.random.default_rng(seed); atr=[]; ate=[]
    for a in actors:
        ids=np.flatnonzero(actor==a).copy(); rng.shuffle(ids)
        if len(ids)<2:
            continue
        cut=max(1,min(len(ids)-1,len(ids)//2)); atr.extend(ids[:cut]); ate.extend(ids[cut:])
    atr=np.asarray(atr,dtype=int); ate=np.asarray(ate,dtype=int)
    attacker=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,class_weight="balanced",random_state=seed))
    attacker.fit(X[atr],actor[atr]); proba=attacker.predict_proba(X[ate]); classes=attacker.classes_
    auc,auc_lo,auc_hi=bootstrap_auc(actor[ate],proba,classes,seed+2)

    result={
      "representation":"motion-features-v1","clips":len(files),"actors":len(actors),"feature_dim":int(X.shape[1]),
      "utility_protocol":f"{folds}-fold actor-disjoint cross-validation",
      "emotion_macro_f1":f1,"emotion_macro_f1_ci95":[f1_lo,f1_hi],"utility_folds":fold_rows,
      "privacy_protocol":"adaptive closed-set identity attacker retrained on released non-image features; clip-disjoint per actor",
      "identity_inference_auc":auc,"identity_inference_auc_ci95":[auc_lo,auc_hi],
      "interpretation":"Identity AUC nearer 0.5 is better. Report privacy and utility together. Do not claim anonymity or clinical validity from this experiment alone."
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/cremad_minimum_representation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2)); return result

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--folds",type=int,default=5)
    a=ap.parse_args(); run(a.root,a.folds)
