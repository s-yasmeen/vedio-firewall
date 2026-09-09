"""TAPF-MIN v3.1 FER repair benchmark on CREMA-D.

Development-only ablation:
A. face-focused RGB only, no identity adversary
B. face-focused RGB + learned motion residuals, no identity adversary
C. face-focused RGB + motion + delayed identity suppression

Actor-disjoint folds are mandatory. This runner is for architecture selection only; a
separate frozen-model held-out privacy audit is required before any final claim.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, random, re
import cv2, numpy as np, torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader

from tapf.face_preprocess import FacePreprocessor
from tapf.fer_v31 import FERV31Config, TAPFMinFERV31

NAME_RE=re.compile(r'^(\d{4})_DFA_(ANG|DIS|FEA|HAP|NEU|SAD)_XX\.flv$',re.I)
EMOTIONS=np.asarray(['ANG','DIS','FEA','HAP','NEU','SAD'])
EMO_TO_I={e:i for i,e in enumerate(EMOTIONS)}
IMAGENET_MEAN=np.asarray([.485,.456,.406],np.float32)
IMAGENET_STD=np.asarray([.229,.224,.225],np.float32)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def sample_rgb(path, n):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError(f'cannot open {path}')
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ids=np.linspace(0,max(0,count-1),num=n).round().astype(int)
    frames=[]
    for i in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,bgr=cap.read()
        if not ok: cap.release(); raise RuntimeError(f'frame read failed: {path}')
        frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release(); return frames


class FaceVideoDataset(Dataset):
    def __init__(self,records,actor_to_i,frames=8,size=96,augment=False,seed=42):
        self.records=list(records); self.actor_to_i=actor_to_i; self.frames=frames; self.size=size
        self.augment=augment; self.seed=seed; self.face=FacePreprocessor(output_size=size,margin=.20)
    def __len__(self): return len(self.records)
    def __getitem__(self,idx):
        path,actor,emo,orig=self.records[idx]
        raw=sample_rgb(path,self.frames)
        crops,det,ali,valid,reused=self.face.process_sequence(raw)
        x=crops.astype(np.float32)/255.0
        x=(x-IMAGENET_MEAN)/IMAGENET_STD
        if self.augment:
            rng=np.random.default_rng(self.seed+int(orig))
            if rng.random()<.5: x=x[:,:,::-1,:].copy()
        x=torch.from_numpy(x).permute(0,3,1,2)
        ident=-1 if self.actor_to_i is None else self.actor_to_i[actor]
        return (
            x, torch.tensor(EMO_TO_I[emo]), torch.tensor(ident), int(orig),
            torch.tensor(det.mean(),dtype=torch.float32),
            torch.tensor(ali.mean(),dtype=torch.float32),
            torch.tensor(valid,dtype=torch.bool),
            torch.tensor(reused.mean(),dtype=torch.float32),
        )


class RGBOnlyWrapper(torch.nn.Module):
    def __init__(self,model): super().__init__(); self.model=model
    def forward(self,frames,identity_grl_strength=0.0,mask=None):
        rgb=self.model.encode_rgb(frames)
        motion=torch.zeros(frames.shape[0],frames.shape[1],self.model.config.motion_dim,device=frames.device)
        fused,gate=self.model.fusion(rgb,motion)
        seq,_=self.model.temporal(fused); pooled,attn=self.model.attn(seq,mask=mask)
        latent=self.model.task_projection(pooled)
        return {'emotion_logits':self.model.emotion_head(latent),'identity_logits':self.model.identity_head(latent),'task_latent':latent,'temporal_attention':attn,'fusion_gate':gate}


def train_fold(train_records,test_records,*,variant,epochs,batch_size,frames,size,seed,device,pretrained):
    actors=sorted({r[1] for r in train_records}); actor_to_i={a:i for i,a in enumerate(actors)}
    tr=FaceVideoDataset(train_records,actor_to_i,frames,size,True,seed)
    te=FaceVideoDataset(test_records,None,frames,size,False,seed)
    trl=DataLoader(tr,batch_size=batch_size,shuffle=True,num_workers=0)
    tel=DataLoader(te,batch_size=batch_size,shuffle=False,num_workers=0)
    cfg=FERV31Config(n_identities=len(actors),pretrained=pretrained,temporal_hidden=192,motion_dim=96,
                     identity_weight=.10,grl_start_fraction=.35,grl_max_strength=.60)
    core=TAPFMinFERV31(cfg).to(device)
    model=RGBOnlyWrapper(core).to(device) if variant=='rgb' else core
    core.freeze_rgb_backbone()
    opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=2e-3,weight_decay=1e-4)
    history=[]
    for epoch in range(epochs):
        if epoch==max(1,epochs//3):
            core.unfreeze_rgb_tail(4)
            opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=3e-4,weight_decay=1e-4)
        model.train(); losses=[]
        progress=(epoch+1)/epochs
        grl=0.0 if variant!='motion_grl' else core.scheduled_grl(progress,cfg)
        id_weight=0.0 if variant!='motion_grl' else cfg.identity_weight
        for x,y,i,_,_,_,valid,_ in trl:
            x=x.to(device); y=y.to(device); i=i.to(device); valid=valid.to(device)
            # All-invalid sequences contain no trusted face evidence and must not train the model.
            keep=valid.any(dim=1)
            if not bool(keep.any()):
                continue
            x=x[keep]; y=y[keep]; i=i[keep]; valid=valid[keep]
            opt.zero_grad(set_to_none=True)
            out=model(x,identity_grl_strength=grl,mask=valid)
            emo=F.cross_entropy(out['emotion_logits'],y,label_smoothing=.05)
            ident=F.cross_entropy(out['identity_logits'],i)
            loss=emo+id_weight*ident
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(float(loss.detach().cpu()))
        if not losses:
            raise RuntimeError('no valid face sequences available for training fold')
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'grl_strength':float(grl),'identity_weight':float(id_weight)})
    model.eval(); rows=[]; ids=[]; det=[]; ali=[]; val=[]; reuse=[]
    with torch.no_grad():
        for x,_,_,orig,dcov,acov,valid,rcov in tel:
            x=x.to(device); valid=valid.to(device)
            out=model(x,identity_grl_strength=0.0,mask=valid)
            p=torch.softmax(out['emotion_logits'],dim=-1).cpu().numpy()
            rows.append(p); ids.extend(np.asarray(orig).astype(int).tolist())
            det.extend(np.asarray(dcov).astype(float).tolist()); ali.extend(np.asarray(acov).astype(float).tolist())
            val.extend(valid.float().mean(dim=1).cpu().numpy().astype(float).tolist()); reuse.extend(np.asarray(rcov).astype(float).tolist())
    return np.vstack(rows),np.asarray(ids),history,{
        'face_detection_rate':float(np.mean(det)), 'eye_alignment_rate':float(np.mean(ali)),
        'valid_face_rate':float(np.mean(val)), 'bbox_reuse_rate':float(np.mean(reuse)),
    }


def run(root,folds=3,epochs=8,batch_size=8,frames=8,size=96,seed=42,pretrained=True):
    seed_all(seed); root=Path(root); raw=[]
    for p in sorted(root.glob('*.flv')):
        m=NAME_RE.match(p.name)
        if m and p.stat().st_size>1000: raw.append((p,m.group(1),m.group(2).upper()))
    actor=np.asarray([a for _,a,_ in raw]); emotion=np.asarray([e for _,_,e in raw])
    if len(raw)<120 or len(np.unique(actor))<8: raise RuntimeError('insufficient CREMA-D cohort')
    records=[(p,a,e,i) for i,(p,a,e) in enumerate(raw)]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results={}
    for variant in ['rgb','motion','motion_grl']:
        P=np.zeros((len(records),6),np.float32); fold_rows=[]; quality=[]
        splitter=GroupKFold(n_splits=min(folds,len(np.unique(actor))))
        for f,(tr,te) in enumerate(splitter.split(np.arange(len(records)),emotion,groups=actor),1):
            Pf,ids,h,q=train_fold([records[i] for i in tr],[records[i] for i in te],variant=variant,epochs=epochs,batch_size=batch_size,frames=frames,size=size,seed=seed+f*101,device=device,pretrained=pretrained)
            P[ids]=Pf; pred=EMOTIONS[np.argmax(Pf,axis=1)]; quality.append(q)
            fold_rows.append({'fold':f,'accuracy':float(accuracy_score(emotion[te],pred)),'macro_f1':float(f1_score(emotion[te],pred,average='macro',zero_division=0)),'actor_overlap':int(len(set(actor[tr])&set(actor[te]))),'preprocessing_quality':q,'history':h})
        pred=EMOTIONS[np.argmax(P,axis=1)]
        qmean={k:float(np.mean([q[k] for q in quality])) for k in quality[0]}
        results[variant]={'accuracy':float(accuracy_score(emotion,pred)),'macro_f1':float(f1_score(emotion,pred,average='macro',zero_division=0)),'preprocessing_quality_mean':qmean,'folds':fold_rows}
    out={'dataset':'CREMA-D DFA cohort supplied to runner','scope':'development architecture selection only; not final validation','variants':results,'selection_warning':'Do not report the selected variant OOF score as final evidence. Freeze the architecture, then use a separate held-out actor cohort and one frozen model for final utility/privacy audit.','parameters':{'folds':folds,'epochs':epochs,'batch_size':batch_size,'frames':frames,'size':size,'seed':seed,'pretrained':pretrained}}
    Path('results').mkdir(exist_ok=True); Path('results/cremad_v31_motion_fer.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2)); return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--folds',type=int,default=3); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--frames',type=int,default=8); ap.add_argument('--size',type=int,default=96); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--no-pretrained',action='store_true'); a=ap.parse_args()
    run(a.root,a.folds,a.epochs,a.batch_size,a.frames,a.size,a.seed,not a.no_pretrained)
