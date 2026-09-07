"""Actor-disjoint CREMA-D video benchmark for TAPF-MIN.

Bounded DFA subset. Measures emotion utility, frame/sequence identity leakage, temporal
leakage, and processing latency. The HOG attacker remains a lightweight baseline; modern
FaceNet/ArcFace experiments are separate. Outputs include conservative bootstrap bounds
for the deployment gate. No result is clinical validation.
"""
from pathlib import Path
import argparse, json, re, time
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from skimage.feature import hog
from tapf.transforms import apply_transform

EMOTIONS = ["ANG","DIS","FEA","HAP","NEU","SAD"]
ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]
NAME_RE = re.compile(r"^(\d{4})_DFA_(ANG|DIS|FEA|HAP|NEU|SAD)_XX\.flv$", re.I)
MIN_REAL_VIDEO_BYTES = 1000


def desc(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); gray=cv2.resize(gray,(96,96))
    v=hog(gray,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)
    n=np.linalg.norm(v); return v/n if n else v


def read_clip(path, alpha, every=8, max_frames=24):
    cap=cv2.VideoCapture(str(path)); vecs=[]; middle=None; i=0; latencies=[]
    while True:
        ok,frame=cap.read()
        if not ok: break
        if i%every==0:
            t0=time.perf_counter()
            protected=frame if alpha==0 else apply_transform("static_tapf",frame,alpha)
            v=desc(protected)
            latencies.append((time.perf_counter()-t0)*1000.0)
            vecs.append(v)
            if middle is None or len(vecs)==2: middle=v
            if len(vecs)>=max_frames: break
        i+=1
    cap.release()
    if not vecs: return None,None,[]
    seq=np.mean(np.asarray(vecs),axis=0); n=np.linalg.norm(seq)
    if n: seq=seq/n
    return middle,seq,latencies


def pairs(labels, seed=42, max_each=1200):
    rng=np.random.default_rng(seed); labels=np.asarray(labels); by={k:np.flatnonzero(labels==k) for k in np.unique(labels)}
    g=[]
    for ids in by.values():
        if len(ids)>=2:
            for _ in range(min(20,len(ids)*3)):
                a,b=rng.choice(ids,2,replace=False); g.append((int(a),int(b),1))
    rng.shuffle(g); g=g[:max_each]; keys=list(by); imp=[]
    while len(imp)<min(max_each,len(g)):
        ka,kb=rng.choice(keys,2,replace=False); imp.append((int(rng.choice(by[ka])),int(rng.choice(by[kb])),0))
    return g+imp


def pair_scores(X, ps):
    y=np.asarray([p[2] for p in ps],dtype=int)
    s=np.asarray([float(np.dot(X[a],X[b])) for a,b,_ in ps],dtype=float)
    return y,s


def auc_with_ci(y,s,seed=42,resamples=500):
    auc=float(roc_auc_score(y,s)); rng=np.random.default_rng(seed); vals=[]; n=len(y)
    for _ in range(resamples):
        idx=rng.integers(0,n,n); yy=y[idx]
        if np.unique(yy).size<2: continue
        vals.append(roc_auc_score(yy,s[idx]))
    lo,hi=np.quantile(vals,[.025,.975]) if vals else (auc,auc)
    return auc,float(lo),float(hi)


def f1_with_ci(y,pred,seed=43,resamples=500):
    val=float(f1_score(y,pred,average="macro",zero_division=0)); rng=np.random.default_rng(seed); vals=[]; n=len(y)
    for _ in range(resamples):
        idx=rng.integers(0,n,n)
        vals.append(f1_score(y[idx],pred[idx],average="macro",zero_division=0))
    lo,hi=np.quantile(vals,[.025,.975])
    return val,float(lo),float(hi)


def risk_from_auc(auc): return float(np.clip((auc-.5)/.5,0,1))


def run(root, train_actors=16, seed=42):
    root=Path(root); files=[]; skipped_pointers=0
    for p in sorted(root.glob("*.flv")):
        m=NAME_RE.match(p.name)
        if not m: continue
        if p.stat().st_size<=MIN_REAL_VIDEO_BYTES: skipped_pointers+=1; continue
        files.append((p,m.group(1),m.group(2)))
    actors=sorted({a for _,a,_ in files})
    print(f"hydrated_clips={len(files)} skipped_pointer_stubs={skipped_pointers}")
    if len(files)<120: raise RuntimeError(f"Need >=120 hydrated DFA clips; found {len(files)}")
    if len(actors)<8: raise RuntimeError(f"Need >=8 actors; found {len(actors)}")
    train_set=set(actors[:min(train_actors,len(actors)-4)]); test_set=set(actors)-train_set
    print(f"clips={len(files)} actors={len(actors)} train={len(train_set)} test={len(test_set)}")

    results=[]; deployment=[]
    actor_labels=[a for _,a,_ in files]; emotion_labels=np.asarray([e for _,_,e in files])
    ps=pairs(actor_labels,seed=seed)
    train_idx=np.asarray([i for i,(_,a,_) in enumerate(files) if a in train_set])
    test_idx=np.asarray([i for i,(_,a,_) in enumerate(files) if a in test_set])

    for alpha in ALPHAS:
        frame_X=[]; seq_X=[]; keep=[]; all_latency=[]
        for i,(p,a,e) in enumerate(files):
            f,s,lat=read_clip(p,alpha)
            if f is not None: frame_X.append(f); seq_X.append(s); keep.append(i); all_latency.extend(lat)
        if len(keep)!=len(files): raise RuntimeError("Unreadable hydrated clips detected; aborting to preserve split integrity")
        frame_X=np.asarray(frame_X); seq_X=np.asarray(seq_X)
        fy,fs=pair_scores(frame_X,ps); sy,ss=pair_scores(seq_X,ps)
        frame_auc,frame_lo,frame_hi=auc_with_ci(fy,fs,seed=seed)
        seq_auc,seq_lo,seq_hi=auc_with_ci(sy,ss,seed=seed+1)
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2500,class_weight="balanced",random_state=seed))
        clf.fit(seq_X[train_idx],emotion_labels[train_idx]); pred=clf.predict(seq_X[test_idx])
        f1,f1_lo,f1_hi=f1_with_ci(emotion_labels[test_idx],pred,seed=seed+2)
        latency_mean=float(np.mean(all_latency)); latency_p95=float(np.quantile(all_latency,.95))
        identity_risk=risk_from_auc(max(frame_auc,seq_auc))
        identity_upper=risk_from_auc(max(frame_hi,seq_hi))
        temporal_risk=risk_from_auc(seq_auc); temporal_upper=risk_from_auc(seq_hi)
        row={
            "method":"original" if alpha==0 else "static_tapf","alpha":alpha,
            "clips":len(files),"actors":len(actors),"train_actors":len(train_set),"test_actors":len(test_set),
            "emotion_macro_f1":f1,"emotion_macro_f1_ci95":[f1_lo,f1_hi],
            "frame_identity_auc":frame_auc,"frame_identity_auc_ci95":[frame_lo,frame_hi],
            "sequence_identity_auc":seq_auc,"sequence_identity_auc_ci95":[seq_lo,seq_hi],
            "temporal_leakage_gain":float(seq_auc-frame_auc),
            "identity_risk":identity_risk,"identity_risk_upper":identity_upper,
            "temporal_risk":temporal_risk,"temporal_risk_upper":temporal_upper,
            "latency_ms_mean":latency_mean,"latency_ms_p95":latency_p95,
            "protocol":"actor-disjoint emotion utility; pairwise HOG identity verification; bootstrap 95% bounds",
        }
        results.append(row); print(row)
        deployment.append({
            "alpha":alpha,
            "identity_risk":identity_risk,"identity_risk_upper":identity_upper,
            "task_utility":f1,"task_utility_lower":f1_lo,
            "temporal_risk":temporal_risk,"temporal_risk_upper":temporal_upper,
            "latency_ms":latency_p95,
            "evaluator_id":"cremad-hog-logreg-bootstrap-v1",
            "sample_count":len(files),
        })

    out=Path("results"); out.mkdir(exist_ok=True)
    (out/"cremad_empirical_frontier.json").write_text(json.dumps(results,indent=2))
    (out/"cremad_deployment_operating_points.json").write_text(json.dumps(deployment,indent=2))
    return results


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--train-actors",type=int,default=16)
    args=ap.parse_args(); run(args.root,args.train_actors)
