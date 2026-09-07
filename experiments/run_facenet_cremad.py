"""Bounded FaceNet sequence attack on hydrated CREMA-D clips.

Threat model: MTCNN localizes the face on the original sampled frame to obtain a stable
face ROI. The same ROI is transformed at alpha in {0, .5, 1.0} and embedded by a
VGGFace2-pretrained InceptionResnetV1. This is intentionally a stronger location-aware
attacker than relying on face detection after protection.

Reports frame- and sequence-level verification ROC-AUC. Missing detections on original
frames are counted explicitly and excluded from embedding comparisons; they are NOT
silently treated as privacy success.
"""
from pathlib import Path
import argparse, json, re
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from tapf.transforms import static_tapf

NAME_RE = re.compile(r"^(\d{4})_DFA_(ANG|DIS|FEA|HAP|NEU|SAD)_XX\.flv$", re.I)
ALPHAS = [0.0, 0.5, 1.0]
MIN_REAL_VIDEO_BYTES = 1000


def l2(v):
    n=np.linalg.norm(v)
    return v/n if n else v


def face_box(mtcnn, frame):
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(Image.fromarray(rgb))
    if boxes is None or len(boxes)==0: return None
    i=int(np.nanargmax(probs)) if probs is not None else 0
    x1,y1,x2,y2=boxes[i]
    h,w=frame.shape[:2]
    x1=max(0,int(x1)); y1=max(0,int(y1)); x2=min(w,int(x2)); y2=min(h,int(y2))
    if x2-x1<20 or y2-y1<20: return None
    return x1,y1,x2,y2


def embed_crop(model, crop):
    rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
    rgb=cv2.resize(rgb,(160,160),interpolation=cv2.INTER_AREA)
    t=torch.from_numpy(rgb).permute(2,0,1).float()
    t=fixed_image_standardization(t).unsqueeze(0)
    with torch.no_grad():
        v=model(t).cpu().numpy()[0]
    return l2(v.astype(np.float32))


def sample_clip(path, mtcnn, model, alpha, every=15, max_frames=6):
    cap=cv2.VideoCapture(str(path)); embs=[]; attempted=0; detected=0; i=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if i%every==0:
            attempted+=1
            box=face_box(mtcnn,frame)
            if box is not None:
                detected+=1
                x1,y1,x2,y2=box
                crop=frame[y1:y2,x1:x2]
                protected=crop if alpha==0 else static_tapf(crop,alpha)
                embs.append(embed_crop(model,protected))
                if len(embs)>=max_frames: break
        i+=1
    cap.release()
    if not embs: return None,None,attempted,detected
    frame_emb=embs[min(1,len(embs)-1)]
    seq=l2(np.mean(np.asarray(embs),axis=0))
    return frame_emb,seq,attempted,detected


def make_pairs(labels,seed=42,max_each=800):
    rng=np.random.default_rng(seed); labels=np.asarray(labels)
    by={k:np.flatnonzero(labels==k) for k in np.unique(labels)}
    g=[]
    for ids in by.values():
        if len(ids)>=2:
            for _ in range(min(15,len(ids)*2)):
                a,b=rng.choice(ids,2,replace=False); g.append((int(a),int(b),1))
    rng.shuffle(g); g=g[:max_each]
    keys=list(by); imp=[]
    while len(imp)<min(max_each,len(g)):
        ka,kb=rng.choice(keys,2,replace=False)
        imp.append((int(rng.choice(by[ka])),int(rng.choice(by[kb])),0))
    return g+imp


def auc_available(X, labels, ps):
    valid=[]
    for a,b,y in ps:
        if X[a] is not None and X[b] is not None:
            valid.append((float(np.dot(X[a],X[b])),y))
    if len(valid)<20 or len({y for _,y in valid})<2: return None,len(valid)
    return float(roc_auc_score([y for _,y in valid],[s for s,_ in valid])),len(valid)


def run(root,max_actors=12,seed=42):
    root=Path(root)
    files=[]
    actor_order=[]
    for p in sorted(root.glob('*.flv')):
        m=NAME_RE.match(p.name)
        if not m or p.stat().st_size<=MIN_REAL_VIDEO_BYTES: continue
        actor=m.group(1)
        if actor not in actor_order: actor_order.append(actor)
        if actor in actor_order[:max_actors]: files.append((p,actor,m.group(2)))
    allowed=set(actor_order[:max_actors])
    files=[r for r in files if r[1] in allowed]
    if len(allowed)<8 or len(files)<48: raise RuntimeError(f'insufficient hydrated subset actors={len(allowed)} clips={len(files)}')

    device=torch.device('cpu')
    mtcnn=MTCNN(keep_all=True,device=device)
    model=InceptionResnetV1(pretrained='vggface2').eval().to(device)
    labels=[a for _,a,_ in files]; ps=make_pairs(labels,seed=seed)
    results=[]
    for alpha in ALPHAS:
        frame_X=[]; seq_X=[]; attempts=0; detections=0
        for p,_,_ in files:
            f,s,a,d=sample_clip(p,mtcnn,model,alpha)
            frame_X.append(f); seq_X.append(s); attempts+=a; detections+=d
        f_auc,f_pairs=auc_available(frame_X,labels,ps)
        s_auc,s_pairs=auc_available(seq_X,labels,ps)
        row={
            'method':'original' if alpha==0 else 'static_tapf','alpha':alpha,
            'clips':len(files),'actors':len(allowed),
            'frame_identity_auc':f_auc,'sequence_identity_auc':s_auc,
            'temporal_leakage_gain':None if f_auc is None or s_auc is None else float(s_auc-f_auc),
            'detection_rate':float(detections/attempts) if attempts else 0.0,
            'frame_pairs_scored':f_pairs,'sequence_pairs_scored':s_pairs,
            'attacker':'FaceNet InceptionResnetV1 VGGFace2; original-frame MTCNN ROI threat model'
        }
        results.append(row); print(row)
    out=Path('results'); out.mkdir(exist_ok=True)
    (out/'cremad_facenet_attack.json').write_text(json.dumps(results,indent=2))
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--max-actors',type=int,default=12)
    args=ap.parse_args(); run(args.root,args.max_actors)
