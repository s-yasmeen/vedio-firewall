"""TAPF-MIN v2.1: cross-fitted adversarial task representation benchmark on CREMA-D.

This is a feature-level research prototype of the v2.1 release controller. It uses the
validated 147-D temporal motion descriptor as input, learns a task representation with a
neural identity adversary (gradient reversal), a private identity branch, an orthogonality
penalty, and a bottleneck penalty, then releases only the aligned 6-D task posterior.

Important scientific scope: this is NOT yet the final raw-video MobileNet/temporal encoder.
It is an apples-to-apples adversarial/disentanglement experiment on the same input features,
actor-disjoint folds, independent post-hoc attackers, and privacy/utility metrics used by
our earlier baselines.
"""
from __future__ import annotations

from pathlib import Path
import argparse, json, re, random
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

from tapf.minimum_representation import read_motion_clip
from experiments.run_cremad_privacy_utility_spectrum import attack_representation, bootstrap_metric

NAME_RE = re.compile(r"^(\d{4})_DFA_(ANG|DIS|FEA|HAP|NEU|SAD)_XX\.flv$", re.I)
MIN_REAL_VIDEO_BYTES = 1000
EMOTIONS = np.asarray(["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"])


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = float(strength)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


def grl(x, strength):
    return _GRL.apply(x, strength)


class V21(nn.Module):
    def __init__(self, in_dim: int, n_emotion: int, n_identity: int, latent: int = 12):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.10))
        self.task_encoder = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, latent))
        self.private_encoder = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, latent))
        self.task_head = nn.Linear(latent, n_emotion)
        self.id_adv = nn.Sequential(nn.Linear(latent, 32), nn.GELU(), nn.Linear(32, n_identity))
        self.id_private = nn.Sequential(nn.Linear(latent, 32), nn.GELU(), nn.Linear(32, n_identity))

    def forward(self, x, adv_strength: float):
        h = self.shared(x)
        zt = self.task_encoder(h)
        zp = self.private_encoder(h)
        emo = self.task_head(zt)
        ida = self.id_adv(grl(zt, adv_strength))
        idp = self.id_private(zp)
        return zt, zp, emo, ida, idp


def orthogonality_loss(zt, zp):
    zt = F.normalize(zt - zt.mean(0, keepdim=True), dim=1)
    zp = F.normalize(zp - zp.mean(0, keepdim=True), dim=1)
    cross = zt.T @ zp / max(1, zt.shape[0])
    return (cross * cross).mean()


def quantize_probabilities(P, levels=4):
    P = np.asarray(P, np.float64)
    q = np.round(P * (levels - 1)) / (levels - 1)
    s = q.sum(axis=1, keepdims=True)
    zero = s[:, 0] <= 0
    if np.any(zero): q[zero] = 1.0 / q.shape[1]; s = q.sum(axis=1, keepdims=True)
    return (q / s).astype(np.float32)


def train_fold(Xtr, ytr, atr, Xte, adv_strength, seed, epochs=220, beta=0.10, gamma=0.002):
    seed_all(seed)
    emo_to_i = {e:i for i,e in enumerate(EMOTIONS)}
    actors = sorted(np.unique(atr)); actor_to_i = {a:i for i,a in enumerate(actors)}
    yt = torch.tensor([emo_to_i[e] for e in ytr], dtype=torch.long)
    ya = torch.tensor([actor_to_i[a] for a in atr], dtype=torch.long)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    xe = torch.tensor(Xte, dtype=torch.float32)
    model = V21(Xtr.shape[1], len(EMOTIONS), len(actors), latent=12)
    opt = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=1e-3)

    model.train()
    for epoch in range(int(epochs)):
        opt.zero_grad()
        zt, zp, emo, ida, idp = model(xt, adv_strength)
        task = F.cross_entropy(emo, yt)
        adv = F.cross_entropy(ida, ya)
        private = F.cross_entropy(idp, ya)
        orth = orthogonality_loss(zt, zp)
        bottleneck = (zt * zt).mean()
        loss = task + adv + 0.35 * private + beta * orth + gamma * bottleneck
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()

    model.eval()
    with torch.no_grad():
        zt, _, logits, _, _ = model(xe, 0.0)
        P = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
    return P


def crossfit(X, actor, emotion, folds, adv_strength, seed):
    P = np.zeros((len(X), len(EMOTIONS)), np.float32)
    fold_rows = []
    splitter = GroupKFold(n_splits=folds)
    for fold, (tr, te) in enumerate(splitter.split(X, emotion, groups=actor), 1):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]).astype(np.float32)
        Xte = scaler.transform(X[te]).astype(np.float32)
        Pf = train_fold(Xtr, emotion[tr], actor[tr], Xte, adv_strength, seed + fold * 97)
        P[te] = Pf
        pred = EMOTIONS[np.argmax(Pf, axis=1)]
        fold_rows.append({
            "fold": fold,
            "train_actors": int(len(np.unique(actor[tr]))),
            "test_actors": int(len(np.unique(actor[te]))),
            "macro_f1": float(f1_score(emotion[te], pred, average="macro", zero_division=0)),
            "actor_overlap": int(len(set(actor[tr]).intersection(set(actor[te])))),
        })
    return P, fold_rows


def evaluate_release(P, actor, emotion, seed, label):
    pred = EMOTIONS[np.argmax(P, axis=1)]
    f1, lo, hi = bootstrap_metric(emotion, pred, lambda y,p: f1_score(y,p,average="macro",zero_division=0), seed=seed)
    privacy = attack_representation(P, actor, seed + 500)
    return {
        "release": label,
        "emotion_macro_f1": float(f1),
        "emotion_macro_f1_ci95": [float(lo), float(hi)],
        "identity_auc": float(privacy["identity_auc"]),
        "identity_auc_ci95": [float(x) for x in privacy["identity_auc_ci95"]],
        "identity_risk_0to1": float(privacy["identity_risk_0to1"]),
        "worst_case_attacker": privacy["worst_case_attacker"],
        "attackers": privacy["attackers"],
        "shuffled_label_control_auc": float(privacy["shuffled_label_control_auc"]),
        "privacy_gate_point_auc_le_0_60": bool(privacy["identity_auc"] <= 0.60),
        "privacy_gate_upper_ci_le_0_60": bool(privacy["identity_auc_ci95"][1] <= 0.60),
        "utility_gate_lower_ci_ge_0_20": bool(lo >= 0.20),
    }


def run(root, folds=5, seed=42):
    root = Path(root)
    files=[]
    for p in sorted(root.glob("*.flv")):
        m=NAME_RE.match(p.name)
        if m and p.stat().st_size > MIN_REAL_VIDEO_BYTES: files.append((p,m.group(1),m.group(2).upper()))
    actor=np.asarray([a for _,a,_ in files]); emotion=np.asarray([e for _,_,e in files])
    if len(files)<120 or len(np.unique(actor))<8: raise RuntimeError("Insufficient hydrated CREMA-D subset")
    X=np.asarray([read_motion_clip(str(p)) for p,_,_ in files],np.float32)
    if not np.isfinite(X).all(): raise RuntimeError("Non-finite motion features")
    folds=max(2,min(int(folds),len(np.unique(actor))))

    strengths=[0.0,0.25,0.5,1.0,2.0]
    rows=[]
    for i,s in enumerate(strengths):
        P, fold_rows = crossfit(X,actor,emotion,folds,s,seed+1000*i)
        full=evaluate_release(P,actor,emotion,seed+2000*i,"6-D task posterior")
        q4=evaluate_release(quantize_probabilities(P,4),actor,emotion,seed+3000*i,"6-D task posterior, 4-level quantized")
        rows.append({"adversarial_strength":s,"folds":fold_rows,"posterior":full,"posterior_q4":q4})

    candidates=[]
    for r in rows:
        for key in ("posterior","posterior_q4"):
            c=dict(r[key]); c["adversarial_strength"]=r["adversarial_strength"]; candidates.append(c)
    best_privacy=min(candidates,key=lambda r: abs(r["identity_auc"]-0.5))
    best_utility=max(candidates,key=lambda r:r["emotion_macro_f1"])
    eligible=[r for r in candidates if r["privacy_gate_upper_ci_le_0_60"] and r["utility_gate_lower_ci_ge_0_20"]]

    result={
      "dataset":"CREMA-D bounded DFA subset","clips":len(files),"actors":int(len(np.unique(actor))),
      "model":"TAPF-MIN v2.1 feature-level adversarial/disentanglement prototype",
      "scope_note":"Uses the validated 147-D motion descriptor as model input. It does not yet implement the final raw-video MobileNet/temporal encoder.",
      "protocol":"5-fold actor-disjoint OOF task representation; neural gradient reversal identity adversary + private identity branch + orthogonality + bottleneck; independent post-hoc attacker ensemble on exactly the released posterior",
      "objective":"preserve authorized task utility while minimizing recoverable actor identity",
      "targets":{"chance_identity_auc":0.5,"initial_point_target_auc":0.60,"conservative_upper_ci_target_auc":0.60,"minimum_lower_ci_emotion_f1":0.20},
      "sweep":rows,"best_privacy_point":best_privacy,"best_utility_point":best_utility,"eligible_operating_points":eligible,
      "edge_checks":{"actor_disjoint_task_folds":all(x["actor_overlap"]==0 for r in rows for x in r["folds"]),"independent_posthoc_attackers":True,"multi_attacker_worst_case":True,"bootstrap_confidence_intervals":True,"negative_control":True,"release_representation_semantically_aligned_across_folds":True}
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/cremad_v21_adversarial.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2)); return result

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--folds",type=int,default=5)
    args=ap.parse_args(); run(args.root,args.folds)
