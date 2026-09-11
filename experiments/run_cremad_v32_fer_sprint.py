"""TAPF-MIN v3.2 FER sprint benchmark for CREMA-D.

Purpose: maximize honest FER utility first, using the full available visual CREMA-D cohort
rather than the historical DFA-only 144-clip slice.  The privacy/release architecture is
not modified by this runner.

Protocol:
- parse all supported CREMA-D visual clips whose filenames encode actor/emotion;
- actor-disjoint outer folds;
- actor-disjoint inner validation split for early stopping;
- fail-closed face preprocessing with cached crops;
- EfficientNet-B0 RGB + motion + GRU/attention;
- staged unfreezing, class weights, mixed precision, cosine LR;
- macro-F1 is primary selection metric;
- final OOF metric includes abstentions from unusable face sequences;
- actor-cluster bootstrap CI is reported.
"""
from __future__ import annotations

from pathlib import Path
import argparse, hashlib, json, math, random, re
import cv2, numpy as np, torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader

from tapf.face_preprocess import FacePreprocessor
from tapf.fer_v32 import FERV32Config, TAPFMinFERV32

# CREMA-D filenames are actor_sentence_emotion_level.ext; do not restrict sentence to DFA.
NAME_RE = re.compile(
    r"^(?P<actor>\d{4})_(?P<sentence>[A-Z0-9]+)_(?P<emotion>ANG|DIS|FEA|HAP|NEU|SAD)_(?P<level>[A-Z0-9]+)\.(?P<ext>flv|mp4|avi|mov|mkv)$",
    re.I,
)
EMOTIONS = np.asarray(["ANG","DIS","FEA","HAP","NEU","SAD"])
EMO_TO_I = {e:i for i,e in enumerate(EMOTIONS)}
IMAGENET_MEAN = np.asarray([.485,.456,.406], np.float32)
IMAGENET_STD = np.asarray([.229,.224,.225], np.float32)


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def discover_records(root: Path):
    rows=[]
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.stat().st_size <= 1000: continue
        m=NAME_RE.match(p.name)
        if not m: continue
        rows.append((p,m.group("actor"),m.group("emotion").upper()))
    return rows


def sample_rgb(path: Path, n: int):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError(f"cannot open {path}")
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        cap.release(); raise RuntimeError(f"invalid frame count: {path}")
    ids=np.linspace(0,count-1,num=min(n,count)).round().astype(int)
    frames=[]
    for i in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,bgr=cap.read()
        if not ok or bgr is None:
            cap.release(); raise RuntimeError(f"frame read failed: {path} @ {i}")
        frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release()
    # pad short clips by repeating final frame to preserve T
    while len(frames)<n: frames.append(frames[-1].copy())
    return frames


def cache_key(path: Path, frames: int, size: int):
    s=f"{path.resolve()}|{path.stat().st_size}|{int(path.stat().st_mtime)}|{frames}|{size}"
    return hashlib.sha1(s.encode()).hexdigest()


class CachedFaceDataset(Dataset):
    def __init__(self, records, actor_to_i, *, cache_dir, frames=16, size=128, augment=False, seed=42):
        self.records=list(records); self.actor_to_i=actor_to_i; self.frames=frames; self.size=size
        self.cache_dir=Path(cache_dir); self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.augment=augment; self.seed=seed
        self.face=FacePreprocessor(output_size=size,margin=.20,max_reuse_frames=2)

    def __len__(self): return len(self.records)

    def _load_or_build(self,path):
        cp=self.cache_dir/(cache_key(path,self.frames,self.size)+".npz")
        if cp.exists():
            z=np.load(cp)
            return z["crops"],z["det"],z["ali"],z["valid"],z["reused"]
        crops,det,ali,valid,reused=self.face.process_sequence(sample_rgb(path,self.frames))
        np.savez_compressed(cp,crops=crops,det=det,ali=ali,valid=valid,reused=reused)
        return crops,det,ali,valid,reused

    def __getitem__(self,idx):
        path,actor,emo,orig=self.records[idx]
        crops,det,ali,valid,reused=self._load_or_build(path)
        x=crops.astype(np.float32)/255.0
        if self.augment:
            rng=np.random.default_rng(self.seed+int(orig))
            if rng.random()<.5: x=x[:,:,::-1,:].copy()
            # mild photometric jitter only; preserve expression geometry
            gain=float(rng.uniform(.90,1.10)); bias=float(rng.uniform(-.04,.04))
            x=np.clip(x*gain+bias,0,1)
        x=(x-IMAGENET_MEAN)/IMAGENET_STD
        ident=-1 if self.actor_to_i is None else self.actor_to_i[actor]
        return (
            torch.from_numpy(x).permute(0,3,1,2),
            torch.tensor(EMO_TO_I[emo]), torch.tensor(ident), torch.tensor(int(orig)),
            torch.tensor(valid,dtype=torch.bool), torch.tensor(float(np.mean(det))),
            torch.tensor(float(np.mean(ali))), torch.tensor(float(np.mean(reused))),
        )


def class_weights(records,device):
    counts=np.zeros(len(EMOTIONS),dtype=np.float64)
    for _,_,e,_ in records: counts[EMO_TO_I[e]]+=1
    w=counts.sum()/np.maximum(counts,1.0)
    w=w/w.mean()
    return torch.tensor(w,dtype=torch.float32,device=device),counts.astype(int).tolist()


def evaluate(model,loader,device):
    model.eval(); probs=[]; ys=[]; ids=[]; usable_all=[]; quality=[]
    with torch.no_grad():
        for x,y,_,orig,valid,det,ali,reused in loader:
            x=x.to(device); valid=valid.to(device)
            usable=valid.any(dim=1)
            p=np.full((len(y),6),1/6,dtype=np.float32)
            if bool(usable.any()):
                out=model(x[usable],identity_grl_strength=0.0,mask=valid[usable])
                p[np.asarray(usable.cpu())]=torch.softmax(out["emotion_logits"],dim=-1).cpu().numpy()
            probs.append(p); ys.extend(np.asarray(y)); ids.extend(np.asarray(orig)); usable_all.extend(np.asarray(usable.cpu()))
            vr=valid.float().mean(dim=1).cpu().numpy()
            for j in range(len(y)):
                quality.append({"id":int(orig[j]),"valid_face_rate":float(vr[j]),"face_detection_rate":float(det[j]),"eye_alignment_rate":float(ali[j]),"bbox_reuse_rate":float(reused[j]),"usable":bool(usable[j])})
    P=np.vstack(probs); y=np.asarray(ys,dtype=int); usable=np.asarray(usable_all,dtype=bool)
    pred=np.full(len(y),-1,dtype=int); pred[usable]=np.argmax(P[usable],axis=1)
    # score unusable sequences as incorrect without inventing a class
    macro=float(f1_score(y,pred,labels=np.arange(6),average="macro",zero_division=0))
    acc=float(np.mean(pred==y))
    return {"probs":P,"y":y,"ids":np.asarray(ids,dtype=int),"usable":usable,"macro_f1":macro,"accuracy":acc,"quality":quality}


def train_one(train_records,val_records,test_records,*,cache_dir,frames,size,epochs,batch_size,seed,device,use_grl=False):
    actors=sorted({r[1] for r in train_records}); actor_to_i={a:i for i,a in enumerate(actors)}
    tr=CachedFaceDataset(train_records,actor_to_i,cache_dir=cache_dir,frames=frames,size=size,augment=True,seed=seed)
    va=CachedFaceDataset(val_records,None,cache_dir=cache_dir,frames=frames,size=size,augment=False,seed=seed)
    te=CachedFaceDataset(test_records,None,cache_dir=cache_dir,frames=frames,size=size,augment=False,seed=seed)
    trl=DataLoader(tr,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=torch.cuda.is_available(),persistent_workers=False)
    val=DataLoader(va,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=torch.cuda.is_available())
    tel=DataLoader(te,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=torch.cuda.is_available())
    cfg=FERV32Config(n_identities=len(actors),pretrained=True)
    model=TAPFMinFERV32(cfg).to(device); model.freeze_rgb_backbone()
    weights,counts=class_weights(train_records,device)
    opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=1.5e-3,weight_decay=2e-4)
    scaler=torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best=-1.0; best_state=None; patience=5; stale=0; history=[]
    sched=None

    for epoch in range(epochs):
        if epoch==2:
            model.unfreeze_rgb_tail(3)
            opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=2.5e-4,weight_decay=2e-4)
            sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs-2),eta_min=2e-6)
        model.train(); losses=[]
        progress=(epoch+1)/epochs
        grl=model.scheduled_grl(progress,cfg) if use_grl else 0.0
        id_weight=cfg.identity_weight if use_grl else 0.0
        for x,y,ident,_,valid,_,_,_ in trl:
            x=x.to(device,non_blocking=True); y=y.to(device); ident=ident.to(device); valid=valid.to(device)
            keep=valid.any(dim=1)
            if not bool(keep.any()): continue
            x=x[keep]; y=y[keep]; ident=ident[keep]; valid=valid[keep]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out=model(x,identity_grl_strength=grl,mask=valid)
                emo=F.cross_entropy(out["emotion_logits"],y,weight=weights,label_smoothing=.03)
                ident_loss=F.cross_entropy(out["identity_logits"],ident)
                loss=emo+id_weight*ident_loss
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        if sched is not None: sched.step()
        ev=evaluate(model,val,device)
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)) if losses else math.nan,"val_macro_f1":ev["macro_f1"],"val_accuracy":ev["accuracy"],"grl":float(grl),"lr":float(opt.param_groups[0]["lr"])})
        if ev["macro_f1"]>best+1e-4:
            best=ev["macro_f1"]; stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=patience and epoch>=7: break
    if best_state is None: raise RuntimeError("no valid checkpoint")
    model.load_state_dict(best_state)
    test=evaluate(model,tel,device)
    return model,test,history,counts,best


def actor_cluster_bootstrap(y,pred,actors,n_boot=1000,seed=123):
    rng=np.random.default_rng(seed); ua=np.unique(actors); vals=[]
    for _ in range(n_boot):
        sampled=rng.choice(ua,size=len(ua),replace=True)
        idx=np.concatenate([np.where(actors==a)[0] for a in sampled])
        vals.append(f1_score(y[idx],pred[idx],labels=np.arange(6),average="macro",zero_division=0))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def run(root,*,folds=3,epochs=18,batch_size=6,frames=16,size=128,seed=42,cache_dir="/kaggle/working/tapf_face_cache",use_grl=False):
    seed_all(seed); root=Path(root); raw=discover_records(root)
    actor=np.asarray([a for _,a,_ in raw]); emo=np.asarray([e for _,_,e in raw])
    if len(raw)<120 or len(np.unique(actor))<8: raise RuntimeError(f"insufficient cohort: {len(raw)} clips / {len(np.unique(actor))} actors")
    records=[(p,a,e,i) for i,(p,a,e) in enumerate(raw)]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splitter=GroupKFold(n_splits=min(folds,len(np.unique(actor))))
    P=np.zeros((len(records),6),np.float32); usable=np.zeros(len(records),dtype=bool); fold_info=[]; all_quality=[]

    for f,(tr_idx,te_idx) in enumerate(splitter.split(np.arange(len(records)),emo,groups=actor),1):
        # actor-disjoint inner validation split from outer-train actors
        gss=GroupShuffleSplit(n_splits=1,test_size=.18,random_state=seed+f*17)
        inner_train_rel,val_rel=next(gss.split(tr_idx,groups=actor[tr_idx]))
        fit_idx=tr_idx[inner_train_rel]; va_idx=tr_idx[val_rel]
        assert not (set(actor[fit_idx])&set(actor[va_idx]) or set(actor[tr_idx])&set(actor[te_idx]))
        _,ev,h,counts,best=train_one(
            [records[i] for i in fit_idx],[records[i] for i in va_idx],[records[i] for i in te_idx],
            cache_dir=cache_dir,frames=frames,size=size,epochs=epochs,batch_size=batch_size,
            seed=seed+f*101,device=device,use_grl=use_grl)
        P[ev["ids"]]=ev["probs"]; usable[ev["ids"]]=ev["usable"]; all_quality.extend(ev["quality"])
        fold_info.append({"fold":f,"train_actors":len(set(actor[fit_idx])),"val_actors":len(set(actor[va_idx])),"test_actors":len(set(actor[te_idx])),"test_macro_f1":ev["macro_f1"],"test_accuracy":ev["accuracy"],"best_val_macro_f1":best,"class_counts_train":counts,"history":h})
        print(f"fold {f}: test macro-F1={ev['macro_f1']:.4f} acc={ev['accuracy']:.4f} best-val={best:.4f}")

    y=np.asarray([EMO_TO_I[e] for e in emo]); pred=np.full(len(records),-1,dtype=int); pred[usable]=np.argmax(P[usable],axis=1)
    macro=float(f1_score(y,pred,labels=np.arange(6),average="macro",zero_division=0)); acc=float(np.mean(pred==y))
    ci=actor_cluster_bootstrap(y,pred,actor,n_boot=1000,seed=seed+999)
    q={k:float(np.mean([r[k] for r in all_quality])) for k in ["valid_face_rate","face_detection_rate","eye_alignment_rate","bbox_reuse_rate"]}
    q["usable_sequence_rate"]=float(np.mean([r["usable"] for r in all_quality]))
    cm=confusion_matrix(y,pred,labels=np.arange(6)).tolist()
    out={
        "dataset":"CREMA-D visual cohort discovered from supplied root",
        "clips":len(records),"actors":int(len(np.unique(actor))),
        "all_sentences_included":True,"historical_dfa_only_restriction_removed":True,
        "actor_disjoint":True,"inner_actor_disjoint_validation":True,
        "model":"EfficientNet-B0 + motion residual CNN + GRU + temporal attention",
        "identity_adversary_enabled":bool(use_grl),
        "macro_f1_with_abstention_penalty":macro,"macro_f1_actor_cluster_ci95":ci,
        "accuracy_with_abstention_penalty":acc,"abstention_rate":float(1-np.mean(usable)),
        "preprocessing_quality":q,"confusion_matrix_labels":EMOTIONS.tolist(),"confusion_matrix":cm,
        "folds":fold_info,
        "parameters":{"folds":folds,"epochs":epochs,"batch_size":batch_size,"frames":frames,"size":size,"seed":seed},
        "gate_note":"Utility sprint only. Passing FER does not by itself pass the final privacy gate; freeze the selected model and run fixed-model privacy audit next.",
    }
    Path("results").mkdir(exist_ok=True)
    out_path=Path("results/cremad_v32_fer_sprint.json"); out_path.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:out[k] for k in ["clips","actors","macro_f1_with_abstention_penalty","macro_f1_actor_cluster_ci95","accuracy_with_abstention_penalty","abstention_rate","preprocessing_quality"]},indent=2))
    print("saved",out_path)
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("root")
    ap.add_argument("--folds",type=int,default=3); ap.add_argument("--epochs",type=int,default=18)
    ap.add_argument("--batch-size",type=int,default=6); ap.add_argument("--frames",type=int,default=16)
    ap.add_argument("--size",type=int,default=128); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--cache-dir",default="/kaggle/working/tapf_face_cache"); ap.add_argument("--use-grl",action="store_true")
    a=ap.parse_args(); run(a.root,folds=a.folds,epochs=a.epochs,batch_size=a.batch_size,frames=a.frames,size=a.size,seed=a.seed,cache_dir=a.cache_dir,use_grl=a.use_grl)
