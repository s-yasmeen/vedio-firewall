"""TAPF-MIN v2.4 composition-aware identity-invariant benchmark.

Prospective experiment. It does not rewrite prior v2.1-v2.3 evidence.

New over v2.3:
- task-conditioned actor-output distribution matching, aimed directly at repeated-release leakage,
- dual adversaries: clip-level identity and actor-aggregate identity,
- smaller stochastic latent bottleneck,
- task-only deterministic release as the primary mechanism,
- session-stable categorical release: repeated requests reuse one approved task output,
- frozen v2.2 chance-centered privacy gate, unchanged.

Research boundary: empirical benchmark only; not formal anonymity or differential privacy.
"""
from __future__ import annotations

from pathlib import Path
import argparse, json, random, re
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

from tapf.minimum_representation import read_motion_clip
from tapf.privacy_amplification import hard_task_release
from tapf.privacy_gate_v22 import effective_auc, evaluate_v22_gate
from experiments.run_cremad_final_validation import _attack, repeated_release_attack
from experiments.run_cremad_privacy_utility_spectrum import bootstrap_metric

NAME_RE = re.compile(r"^(\d{4})_DFA_(ANG|DIS|FEA|HAP|NEU|SAD)_XX\.flv$", re.I)
MIN_REAL_VIDEO_BYTES = 1000
EMOTIONS = np.asarray(["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"])


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)


class V24Net(nn.Module):
    def __init__(self, in_dim: int, n_task: int, n_id: int, latent: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(96, 40), nn.GELU(),
        )
        self.encoder = nn.Linear(40, latent)
        self.task_head = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, n_task))
        self.clip_adv = nn.Sequential(nn.Linear(latent, 24), nn.GELU(), nn.Linear(24, n_id))
        self.aggregate_adv = nn.Sequential(nn.Linear(n_task, 24), nn.GELU(), nn.Linear(24, n_id))

    def encode(self, x, noise_std: float = 0.0):
        z = self.encoder(self.backbone(x))
        z = F.layer_norm(z, (z.shape[-1],))
        if self.training and noise_std > 0:
            z = z + torch.randn_like(z) * float(noise_std)
        return z


def uniform_confusion_loss(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    return F.kl_div(logp, torch.full_like(logp, 1.0 / logits.shape[1]), reduction="batchmean")


def actor_center_collapse(z, y_task, y_actor):
    losses=[]
    for t in torch.unique(y_task):
        mt=y_task==t; centers=[]
        for a in torch.unique(y_actor[mt]):
            m=mt & (y_actor==a)
            if int(m.sum())>0: centers.append(z[m].mean(0))
        if len(centers)>=2:
            C=torch.stack(centers); losses.append(((C-C.mean(0,keepdim=True))**2).mean())
    return torch.stack(losses).mean() if losses else z.new_tensor(0.0)


def actor_output_matching(task_logits, y_task, y_actor):
    """Remove actor-specific task-posterior signatures conditional on true task.

    Within each true task class, every actor's mean posterior is pulled toward the
    population mean posterior for that same class. This targets the stable confusion
    patterns that a repeated-release attacker can aggregate across clips.
    """
    P=F.softmax(task_logits,dim=1); losses=[]
    for t in torch.unique(y_task):
        mt=y_task==t
        global_mean=P[mt].mean(0).detach()
        for a in torch.unique(y_actor[mt]):
            m=mt & (y_actor==a)
            if int(m.sum())>0:
                losses.append(F.mse_loss(P[m].mean(0), global_mean))
    return torch.stack(losses).mean() if losses else P.new_tensor(0.0)


def actor_aggregate_views(P, y_actor):
    """Create two deterministic aggregate views per actor for aggregate adversary."""
    views=[]; labels=[]
    for a in torch.unique(y_actor):
        idx=torch.where(y_actor==a)[0]
        if len(idx)<2: continue
        even=idx[::2]; odd=idx[1::2]
        if len(even): views.append(P[even].mean(0)); labels.append(a)
        if len(odd): views.append(P[odd].mean(0)); labels.append(a)
    if not views:
        return P.new_zeros((0,P.shape[1])), y_actor.new_zeros((0,))
    return torch.stack(views), torch.stack(labels)


def train_fold(Xtr,ytr,atr,Xte,seed:int,id_weight:float,center_weight:float,
               output_weight:float,aggregate_weight:float,noise_std:float,epochs:int=300):
    seed_all(seed)
    emo_to_i={e:i for i,e in enumerate(EMOTIONS)}
    actors=sorted(np.unique(atr)); actor_to_i={a:i for i,a in enumerate(actors)}
    xt=torch.tensor(Xtr,dtype=torch.float32); xe=torch.tensor(Xte,dtype=torch.float32)
    yt=torch.tensor([emo_to_i[e] for e in ytr],dtype=torch.long)
    ya=torch.tensor([actor_to_i[a] for a in atr],dtype=torch.long)
    net=V24Net(Xtr.shape[1],len(EMOTIONS),len(actors),latent=4)
    main=list(net.backbone.parameters())+list(net.encoder.parameters())+list(net.task_head.parameters())
    opt_main=torch.optim.AdamW(main,lr=0.003,weight_decay=3e-3)
    opt_adv=torch.optim.AdamW(list(net.clip_adv.parameters())+list(net.aggregate_adv.parameters()),lr=0.003,weight_decay=3e-3)

    for _ in range(int(epochs)):
        # Strengthen both identity attackers against the current frozen representation.
        net.train(); opt_adv.zero_grad()
        with torch.no_grad():
            zd=net.encode(xt,0.0); logits_d=net.task_head(zd); Pd=F.softmax(logits_d,dim=1)
        loss_adv=F.cross_entropy(net.clip_adv(zd.detach()),ya)
        av,al=actor_aggregate_views(Pd.detach(),ya)
        if len(av): loss_adv=loss_adv+F.cross_entropy(net.aggregate_adv(av),al)
        loss_adv.backward(); opt_adv.step()

        for p in list(net.clip_adv.parameters())+list(net.aggregate_adv.parameters()): p.requires_grad_(False)
        opt_main.zero_grad()
        z=net.encode(xt,noise_std); logits=net.task_head(z); P=F.softmax(logits,dim=1)
        task=F.cross_entropy(logits,yt)
        clip_conf=uniform_confusion_loss(net.clip_adv(z))
        av,_=actor_aggregate_views(P,ya)
        agg_conf=uniform_confusion_loss(net.aggregate_adv(av)) if len(av) else z.new_tensor(0.0)
        center=actor_center_collapse(z,yt,ya)
        output_match=actor_output_matching(logits,yt,ya)
        bottleneck=(z*z).mean()
        entropy_floor=torch.relu(0.55-( -(P*torch.log(P+1e-8)).sum(1).mean()/np.log(len(EMOTIONS)) ))
        loss=(task + id_weight*clip_conf + aggregate_weight*agg_conf + center_weight*center
              + output_weight*output_match + 0.003*bottleneck + 0.05*entropy_floor)
        loss.backward(); nn.utils.clip_grad_norm_(main,5.0); opt_main.step()
        for p in list(net.clip_adv.parameters())+list(net.aggregate_adv.parameters()): p.requires_grad_(True)

    net.eval()
    with torch.no_grad():
        z=net.encode(xe,0.0); P=torch.softmax(net.task_head(z),dim=1).cpu().numpy().astype(np.float32)
    return P


def crossfit(X,actor,emotion,folds,cfg,seed):
    P=np.zeros((len(X),len(EMOTIONS)),np.float32); fold_rows=[]
    for fold,(tr,te) in enumerate(GroupKFold(n_splits=folds).split(X,emotion,groups=actor),1):
        scaler=StandardScaler().fit(X[tr])
        Pf=train_fold(scaler.transform(X[tr]).astype(np.float32),emotion[tr],actor[tr],
                      scaler.transform(X[te]).astype(np.float32),seed+fold*131,**cfg)
        P[te]=Pf; pred=EMOTIONS[np.argmax(Pf,axis=1)]
        fold_rows.append({"fold":fold,"macro_f1":float(f1_score(emotion[te],pred,average="macro",zero_division=0)),
                          "train_actors":int(len(np.unique(actor[tr]))),"test_actors":int(len(np.unique(actor[te]))),
                          "actor_overlap":int(len(set(actor[tr]).intersection(set(actor[te]))))})
    return P,fold_rows


def evaluate_release(Z,actor,emotion,seed,label):
    pred=EMOTIONS[np.argmax(Z,axis=1)]
    f1,lo,hi=bootstrap_metric(emotion,pred,lambda y,p:f1_score(y,p,average="macro",zero_division=0),seed=seed,n=400)
    clip=_attack(Z,actor,seed+500); repeated=repeated_release_attack(Z,actor,seed+900)
    gate=evaluate_v22_gate(clip_auc_ci95=clip["identity_auc_ci95"],
        repeated_release_auc=repeated["identity_auc"],task_f1_ci95=[lo,hi],
        attacker_aucs=[float(x["auc"]) for x in clip["attackers"]])
    return {"release":label,"emotion_macro_f1":float(f1),"emotion_macro_f1_ci95":[float(lo),float(hi)],
            "clip_identity":clip,"repeated_release_identity":repeated,
            "clip_effective_auc":float(effective_auc(clip["identity_auc"])),
            "repeated_effective_auc":float(effective_auc(repeated["identity_auc"])),"v22_gate":gate}


def deterministic_task_smoothing(P,alpha:float):
    """Deterministic low-information release; no fresh randomness across requests."""
    hard=hard_task_release(P).astype(np.float32); k=hard.shape[1]
    return ((1.0-float(alpha))*hard + float(alpha)/k).astype(np.float32)


def run(root,folds=5,seed=42):
    root=Path(root); files=[]
    for p in sorted(root.glob("*.flv")):
        m=NAME_RE.match(p.name)
        if m and p.stat().st_size>MIN_REAL_VIDEO_BYTES: files.append((p,m.group(1),m.group(2).upper()))
    actor=np.asarray([a for _,a,_ in files]); emotion=np.asarray([e for _,_,e in files])
    if len(files)<120 or len(np.unique(actor))<8: raise RuntimeError(f"Insufficient hydrated cohort: {len(files)} clips/{len(np.unique(actor))} actors")
    X=np.asarray([read_motion_clip(str(p)) for p,_,_ in files],np.float32)
    if not np.isfinite(X).all(): raise RuntimeError("Non-finite motion features")
    folds=max(2,min(int(folds),len(np.unique(actor))))

    configs=[
      {"id_weight":0.75,"center_weight":0.50,"output_weight":2.0,"aggregate_weight":0.75,"noise_std":0.06},
      {"id_weight":1.25,"center_weight":0.75,"output_weight":4.0,"aggregate_weight":1.25,"noise_std":0.08},
      {"id_weight":2.00,"center_weight":1.00,"output_weight":6.0,"aggregate_weight":2.00,"noise_std":0.10},
      {"id_weight":3.00,"center_weight":1.50,"output_weight":8.0,"aggregate_weight":3.00,"noise_std":0.12},
    ]
    candidates=[]; sweep=[]
    for i,cfg in enumerate(configs):
        P,fold_rows=crossfit(X,actor,emotion,folds,cfg,seed+1000*i)
        releases=[("task-label-onehot",hard_task_release(P))]
        for alpha in (0.10,0.25,0.40,0.55):
            releases.append((f"session-stable-task-a{alpha:.2f}",deterministic_task_smoothing(P,alpha)))
        evals=[]
        for j,(name,Z) in enumerate(releases):
            ev=evaluate_release(Z,actor,emotion,seed+3000*i+100*j,name); evals.append(ev)
            candidates.append({"kind":name,"config":cfg,"release_eligible":bool(ev["v22_gate"]["release_eligible"]),
                "clip_effective_auc_upper":float(ev["v22_gate"]["observed"]["clip_effective_auc_upper"]),
                "repeated_effective_auc":float(ev["repeated_effective_auc"]),
                "task_f1_lower_ci":float(ev["emotion_macro_f1_ci95"][0]),"task_f1":float(ev["emotion_macro_f1"])})
        sweep.append({"config":cfg,"folds":fold_rows,"releases":evals})

    def rank(c):
        return (max(c["clip_effective_auc_upper"],c["repeated_effective_auc"]),-c["task_f1_lower_ci"],-c["task_f1"])
    eligible=[c for c in candidates if c["release_eligible"]]
    result={"dataset":"CREMA-D DFA cohort","model":"TAPF-MIN v2.4 composition-aware identity-invariant experiment",
      "clips":len(files),"actors":int(len(np.unique(actor))),
      "frozen_gate":{"max_effective_auc":0.55,"min_task_f1_lower_ci":0.20},
      "methods":["clip identity adversary","actor-aggregate identity adversary","uniform identity confusion",
        "within-task actor latent collapse","task-conditioned actor output matching","4-D stochastic bottleneck",
        "task-only release","session-stable deterministic release","repeated-release attack"],
      "sweep":sweep,"candidate_table":candidates,"best_overall":min(candidates,key=rank),
      "best_eligible":min(eligible,key=rank) if eligible else None,"eligible_operating_points":eligible,
      "integrity":{"threshold_relaxation":False,"actor_disjoint":True,"chance_centered_auc":True,
        "repeated_release_training_target":True,"fresh_release_randomness":False,"fail_closed":True},
      "claim_boundary":"Prospective research benchmark only; not formal anonymity, differential privacy, clinical validation, diagnostic efficacy, or regulatory evidence."}
    Path("results").mkdir(exist_ok=True); out=Path("results/cremad_v24_composition_aware.json")
    out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--folds",type=int,default=5); ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args(); run(a.root,a.folds,a.seed)
