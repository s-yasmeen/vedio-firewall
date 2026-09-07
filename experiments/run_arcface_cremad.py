"""Bounded ArcFace-family sequence attack using InsightFace on hydrated CREMA-D clips.

Detection failures are reported and never interpreted as privacy success. This experiment
is optional because pretrained model assets have separate licensing/usage terms.
"""
from pathlib import Path
import argparse, json, re
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
from attackers.insightface_attacker import InsightFaceAttacker, InsightFaceConfig
from tapf.transforms import static_tapf

NAME_RE=re.compile(r"^(\d{4})_DFA_(ANG|DIS|FEA|HAP|NEU|SAD)_XX\.flv$",re.I)
ALPHAS=[0.0,0.5,1.0]
MIN_REAL_VIDEO_BYTES=1000


def l2(v):
    n=np.linalg.norm(v); return v/n if n else v


def sample_clip(path, attacker, alpha, every=15, max_frames=6):
    cap=cv2.VideoCapture(str(path)); embs=[]; attempted=0; detected=0; i=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if i%every==0:
            attempted+=1
            protected=frame if alpha==0 else static_tapf(frame,alpha)
            emb=attacker.embedding(protected)
            if emb is not None:
                detected+=1; embs.append(emb)
                if len(embs)>=max_frames: break
        i+=1
    cap.release()
    if not embs: return None,None,attempted,detected
    return embs[min(1,len(embs)-1)],l2(np.mean(np.asarray(embs),axis=0)),attempted,detected


def make_pairs(labels,seed=42,max_each=800):
    rng=np.random.default_rng(seed); labels=np.asarray(labels); by={k:np.flatnonzero(labels==k) for k in np.unique(labels)}
    genuine=[]
    for ids in by.values():
        if len(ids)>=2:
            for _ in range(min(15,len(ids)*2)):
                a,b=rng.choice(ids,2,replace=False); genuine.append((int(a),int(b),1))
    rng.shuffle(genuine); genuine=genuine[:max_each]; keys=list(by); impostor=[]
    while len(impostor)<min(max_each,len(genuine)):
        ka,kb=rng.choice(keys,2,replace=False); impostor.append((int(rng.choice(by[ka])),int(rng.choice(by[kb])),0))
    return genuine+impostor


def auc_available(X, ps):
    valid=[]
    for a,b,y in ps:
        if X[a] is not None and X[b] is not None: valid.append((float(np.dot(X[a],X[b])),y))
    if len(valid)<20 or len({y for _,y in valid})<2: return None,len(valid)
    return float(roc_auc_score([y for _,y in valid],[s for s,_ in valid])),len(valid)


def run(root,max_actors=8,seed=42,model_name='buffalo_l'):
    root=Path(root); files=[]; actor_order=[]
    for p in sorted(root.glob('*.flv')):
        m=NAME_RE.match(p.name)
        if not m or p.stat().st_size<=MIN_REAL_VIDEO_BYTES: continue
        actor=m.group(1)
        if actor not in actor_order: actor_order.append(actor)
        if actor in actor_order[:max_actors]: files.append((p,actor,m.group(2)))
    allowed=set(actor_order[:max_actors]); files=[r for r in files if r[1] in allowed]
    if len(allowed)<6 or len(files)<36: raise RuntimeError(f'insufficient hydrated subset actors={len(allowed)} clips={len(files)}')
    attacker=InsightFaceAttacker(InsightFaceConfig(model_name=model_name))
    labels=[a for _,a,_ in files]; ps=make_pairs(labels,seed=seed); results=[]
    for alpha in ALPHAS:
        frame_X=[]; seq_X=[]; attempts=detections=0
        for p,_,_ in files:
            f,s,a,d=sample_clip(p,attacker,alpha); frame_X.append(f); seq_X.append(s); attempts+=a; detections+=d
        f_auc,f_pairs=auc_available(frame_X,ps); s_auc,s_pairs=auc_available(seq_X,ps)
        row={
            'method':'original' if alpha==0 else 'static_tapf','alpha':alpha,'clips':len(files),'actors':len(allowed),
            'frame_identity_auc':f_auc,'sequence_identity_auc':s_auc,
            'temporal_leakage_gain':None if f_auc is None or s_auc is None else float(s_auc-f_auc),
            'detection_rate':float(detections/attempts) if attempts else 0.0,
            'frame_pairs_scored':f_pairs,'sequence_pairs_scored':s_pairs,
            'attacker':f'InsightFace ArcFace-family model={model_name}; protected-frame detection',
            'detection_failure_policy':'UNSCORABLE_NOT_PRIVACY_SUCCESS'
        }
        results.append(row); print(row)
    out=Path('results'); out.mkdir(exist_ok=True); (out/'cremad_arcface_attack.json').write_text(json.dumps(results,indent=2))
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--max-actors',type=int,default=8); ap.add_argument('--model-name',default='buffalo_l')
    a=ap.parse_args(); run(a.root,a.max_actors,model_name=a.model_name)
