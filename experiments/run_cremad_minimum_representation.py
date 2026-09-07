"""Evaluate a non-image TAPF-MIN motion representation on CREMA-D.

Measures actor-disjoint emotion utility and identity inference directly from the released
motion representation. This is the appropriate privacy test for a non-image release: the
attacker is retrained on exactly what the receiver would obtain.

Do not describe a representation as privacy-preserving unless the measured identity attack
and utility bounds satisfy a predeclared policy.
"""
from pathlib import Path
import argparse, json, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tapf.minimum_representation import read_motion_clip

EMOTIONS=["ANG","DIS","FEA","HAP","NEU","SAD"]
NAME_RE=re.compile(r"^(\d{4})_DFA_(ANG|DIS|FEA|HAP|NEU|SAD)_XX\.flv$",re.I)
MIN_REAL_VIDEO_BYTES=1000


def bootstrap_metric(y, pred_or_score, fn, seed=42, n=500):
    rng=np.random.default_rng(seed); vals=[]; y=np.asarray(y); p=np.asarray(pred_or_score)
    point=float(fn(y,p))
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y)); yy=y[idx]
        try: vals.append(float(fn(yy,p[idx])))
        except ValueError: pass
    lo,hi=np.quantile(vals,[.025,.975]) if vals else (point,point)
    return point,float(lo),float(hi)


def run(root, train_actors=16, seed=42):
    root=Path(root); files=[]
    for p in sorted(root.glob("*.flv")):
        m=NAME_RE.match(p.name)
        if m and p.stat().st_size>MIN_REAL_VIDEO_BYTES:
            files.append((p,m.group(1),m.group(2)))
    actors=sorted({a for _,a,_ in files})
    if len(files)<120 or len(actors)<8:
        raise RuntimeError(f"Need >=120 hydrated clips and >=8 actors; got {len(files)} clips/{len(actors)} actors")
    train=set(actors[:min(train_actors,len(actors)-4)])
    X=np.asarray([read_motion_clip(str(p)) for p,_,_ in files],dtype=np.float32)
    actor=np.asarray([a for _,a,_ in files]); emotion=np.asarray([e for _,_,e in files])
    tr=np.asarray([i for i,a in enumerate(actor) if a in train]); te=np.asarray([i for i,a in enumerate(actor) if a not in train])

    utility=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced",random_state=seed))
    utility.fit(X[tr],emotion[tr]); ep=utility.predict(X[te])
    f1,f1_lo,f1_hi=bootstrap_metric(emotion[te],ep,lambda y,p:f1_score(y,p,average="macro",zero_division=0),seed+1)

    # Closed-set identity inference from the released representation. Split clips per actor
    # so the attacker is trained and tested on every test identity without reusing clips.
    rng=np.random.default_rng(seed); atr=[]; ate=[]
    for a in actors:
        ids=np.flatnonzero(actor==a); rng.shuffle(ids); cut=max(1,len(ids)//2)
        atr.extend(ids[:cut]); ate.extend(ids[cut:])
    atr=np.asarray(atr); ate=np.asarray(ate)
    attacker=make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000,class_weight="balanced",random_state=seed))
    attacker.fit(X[atr],actor[atr])
    proba=attacker.predict_proba(X[ate]); classes=attacker.classes_
    # Macro one-vs-rest AUC; chance is 0.5. This attacks the actual released feature vector.
    ybin=np.column_stack([(actor[ate]==c).astype(int) for c in classes])
    auc,auc_lo,auc_hi=bootstrap_metric(ybin,proba,lambda y,s:roc_auc_score(y,s,average="macro",multi_class="ovr"),seed+2)

    result={
      "representation":"motion-features-v1",
      "clips":len(files),"actors":len(actors),
      "emotion_macro_f1":f1,"emotion_macro_f1_ci95":[f1_lo,f1_hi],
      "identity_inference_auc":auc,"identity_inference_auc_ci95":[auc_lo,auc_hi],
      "protocol":"actor-disjoint emotion utility; attacker retrained on released non-image motion features; bootstrap 95% CI",
      "interpretation":"Lower identity AUC is better; utility must be interpreted against a predeclared task threshold. No anonymity or clinical-validity claim."
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/cremad_minimum_representation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2)); return result

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--train-actors",type=int,default=16)
    a=ap.parse_args(); run(a.root,a.train_actors)
