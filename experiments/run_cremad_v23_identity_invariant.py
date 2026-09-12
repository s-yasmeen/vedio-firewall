"""TAPF-MIN v2.3 experimental identity-invariant benchmark.

Goal: reduce identity advantage toward zero while preserving task utility.
This is a prospective research experiment and does not rewrite v2.1/v2.2 results.

Main changes over v2.2:
- alternating minimax identity-adversary training,
- explicit identity-confusion loss toward a uniform posterior,
- within-emotion actor-center collapse penalty,
- stochastic latent bottleneck,
- task-only release candidates,
- privacy-blanket stochastic release candidates,
- same frozen v2.2 chance-centered gate and repeated-release attack.
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
from tapf.privacy_amplification import hard_task_release, privacy_blanket_sample
from tapf.privacy_gate_v22 import effective_auc, evaluate_v22_gate
from experiments.run_cremad_final_validation import _attack, repeated_release_attack
from experiments.run_cremad_privacy_utility_spectrum import bootstrap_metric

NAME_RE = re.compile(r"^(\d{4})_DFA_(ANG|DIS|FEA|HAP|NEU|SAD)_XX\.flv$", re.I)
MIN_REAL_VIDEO_BYTES = 1000
EMOTIONS = np.asarray(["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"])


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)


class IdentityInvariantNet(nn.Module):
    def __init__(self, in_dim: int, n_task: int, n_id: int, latent: int = 8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(96, 48), nn.GELU(),
        )
        self.encoder = nn.Linear(48, latent)
        self.task_head = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, n_task))
        self.id_adv = nn.Sequential(
            nn.Linear(latent, 32), nn.GELU(), nn.Dropout(0.10), nn.Linear(32, n_id)
        )

    def encode(self, x, noise_std: float = 0.0):
        z = self.encoder(self.backbone(x))
        z = F.layer_norm(z, (z.shape[-1],))
        if self.training and noise_std > 0:
            z = z + torch.randn_like(z) * float(noise_std)
        return z


def uniform_confusion_loss(logits: torch.Tensor) -> torch.Tensor:
    """Make identity adversary posterior maximally uninformative."""
    logp = F.log_softmax(logits, dim=1)
    k = logits.shape[1]
    target = torch.full_like(logp, 1.0 / k)
    return F.kl_div(logp, target, reduction="batchmean")


def actor_center_collapse(z, y_task, y_actor):
    """Within each task class, reduce persistent actor-specific center differences."""
    losses = []
    for t in torch.unique(y_task):
        mask_t = y_task == t
        actors = torch.unique(y_actor[mask_t])
        centers = []
        for a in actors:
            m = mask_t & (y_actor == a)
            if int(m.sum()) >= 1:
                centers.append(z[m].mean(0))
        if len(centers) >= 2:
            C = torch.stack(centers)
            losses.append(((C - C.mean(0, keepdim=True)) ** 2).mean())
    return torch.stack(losses).mean() if losses else z.new_tensor(0.0)


def train_fold(Xtr, ytr, atr, Xte, seed: int, id_weight: float, center_weight: float,
               noise_std: float, epochs: int = 260):
    seed_all(seed)
    emo_to_i = {e:i for i,e in enumerate(EMOTIONS)}
    actors = sorted(np.unique(atr)); actor_to_i = {a:i for i,a in enumerate(actors)}
    xt = torch.tensor(Xtr, dtype=torch.float32)
    xe = torch.tensor(Xte, dtype=torch.float32)
    yt = torch.tensor([emo_to_i[e] for e in ytr], dtype=torch.long)
    ya = torch.tensor([actor_to_i[a] for a in atr], dtype=torch.long)

    net = IdentityInvariantNet(Xtr.shape[1], len(EMOTIONS), len(actors), latent=8)
    enc_params = list(net.backbone.parameters()) + list(net.encoder.parameters()) + list(net.task_head.parameters())
    opt_main = torch.optim.AdamW(enc_params, lr=0.0035, weight_decay=2e-3)
    opt_adv = torch.optim.AdamW(net.id_adv.parameters(), lr=0.0035, weight_decay=2e-3)

    for _ in range(int(epochs)):
        # A) strengthen identity adversary on detached representation.
        net.train(); opt_adv.zero_grad()
        with torch.no_grad():
            z_det = net.encode(xt, noise_std=0.0)
        id_logits = net.id_adv(z_det.detach())
        loss_adv = F.cross_entropy(id_logits, ya)
        loss_adv.backward(); opt_adv.step()

        # B) train representation for task utility while confusing the fixed adversary.
        for p in net.id_adv.parameters(): p.requires_grad_(False)
        opt_main.zero_grad()
        z = net.encode(xt, noise_std=noise_std)
        task_logits = net.task_head(z)
        id_logits_enc = net.id_adv(z)
        task = F.cross_entropy(task_logits, yt)
        confusion = uniform_confusion_loss(id_logits_enc)
        center = actor_center_collapse(z, yt, ya)
        bottleneck = (z * z).mean()
        loss = task + float(id_weight) * confusion + float(center_weight) * center + 0.001 * bottleneck
        loss.backward(); nn.utils.clip_grad_norm_(enc_params, 5.0); opt_main.step()
        for p in net.id_adv.parameters(): p.requires_grad_(True)

    net.eval()
    with torch.no_grad():
        z = net.encode(xe, noise_std=0.0)
        logits = net.task_head(z)
        P = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
    return P


def crossfit(X, actor, emotion, folds, cfg, seed):
    P = np.zeros((len(X), len(EMOTIONS)), np.float32)
    fold_rows = []
    splitter = GroupKFold(n_splits=folds)
    for fold, (tr, te) in enumerate(splitter.split(X, emotion, groups=actor), 1):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]).astype(np.float32)
        Xte = scaler.transform(X[te]).astype(np.float32)
        Pf = train_fold(Xtr, emotion[tr], actor[tr], Xte, seed + fold*103, **cfg)
        P[te] = Pf
        pred = EMOTIONS[np.argmax(Pf, axis=1)]
        fold_rows.append({
            "fold": fold,
            "macro_f1": float(f1_score(emotion[te], pred, average="macro", zero_division=0)),
            "train_actors": int(len(np.unique(actor[tr]))),
            "test_actors": int(len(np.unique(actor[te]))),
            "actor_overlap": int(len(set(actor[tr]).intersection(set(actor[te])))),
        })
    return P, fold_rows


def evaluate_once(Z, actor, emotion, seed: int, label: str):
    pred = EMOTIONS[np.argmax(Z, axis=1)]
    f1, lo, hi = bootstrap_metric(
        emotion, pred,
        lambda y,p: f1_score(y, p, average="macro", zero_division=0),
        seed=seed, n=400,
    )
    clip = _attack(Z, actor, seed + 500)
    repeated = repeated_release_attack(Z, actor, seed + 900)
    gate = evaluate_v22_gate(
        clip_auc_ci95=clip["identity_auc_ci95"],
        repeated_release_auc=repeated["identity_auc"],
        task_f1_ci95=[lo, hi],
        attacker_aucs=[float(x["auc"]) for x in clip["attackers"]],
    )
    return {
        "release": label,
        "emotion_macro_f1": float(f1),
        "emotion_macro_f1_ci95": [float(lo), float(hi)],
        "clip_identity": clip,
        "repeated_release_identity": repeated,
        "clip_effective_auc": float(effective_auc(clip["identity_auc"])),
        "repeated_effective_auc": float(effective_auc(repeated["identity_auc"])),
        "v22_gate": gate,
    }


def evaluate_stochastic(P, actor, emotion, seed, alpha, replicates=7):
    rows = []
    for r in range(replicates):
        Z = privacy_blanket_sample(P, alpha, seed + 10007*r)
        rows.append(evaluate_once(Z, actor, emotion, seed + 20011*r, f"privacy-blanket-a{alpha:.2f}"))
    max_clip = max(float(x["v22_gate"]["observed"]["clip_effective_auc_upper"]) for x in rows)
    max_repeat = max(float(x["repeated_effective_auc"]) for x in rows)
    min_f1lo = min(float(x["emotion_macro_f1_ci95"][0]) for x in rows)
    eligible = all(bool(x["v22_gate"]["release_eligible"]) for x in rows)
    return {
        "release": f"privacy-blanket-a{alpha:.2f}",
        "alpha": float(alpha),
        "replicates": rows,
        "conservative_summary": {
            "max_clip_effective_auc_upper": max_clip,
            "max_repeated_effective_auc": max_repeat,
            "min_task_f1_lower_ci": min_f1lo,
            "decision": "RELEASE" if eligible else "BLOCK",
        },
        "release_eligible_all_replicates": eligible,
    }


def run(root, folds=5, seed=42):
    root = Path(root)
    files=[]
    for p in sorted(root.glob("*.flv")):
        m=NAME_RE.match(p.name)
        if m and p.stat().st_size > MIN_REAL_VIDEO_BYTES:
            files.append((p,m.group(1),m.group(2).upper()))
    actor=np.asarray([a for _,a,_ in files]); emotion=np.asarray([e for _,_,e in files])
    if len(files)<120 or len(np.unique(actor))<8:
        raise RuntimeError(f"Insufficient hydrated cohort: {len(files)} clips/{len(np.unique(actor))} actors")
    X=np.asarray([read_motion_clip(str(p)) for p,_,_ in files],np.float32)
    if not np.isfinite(X).all(): raise RuntimeError("Non-finite motion features")
    folds=max(2,min(int(folds),len(np.unique(actor))))

    # Predeclared configurations: progressively stronger identity invariance.
    configs = [
        {"id_weight":0.25,"center_weight":0.10,"noise_std":0.03},
        {"id_weight":0.50,"center_weight":0.25,"noise_std":0.05},
        {"id_weight":1.00,"center_weight":0.50,"noise_std":0.08},
        {"id_weight":1.50,"center_weight":0.75,"noise_std":0.10},
    ]
    candidates=[]; sweep=[]
    for i,cfg in enumerate(configs):
        P, fold_rows = crossfit(X, actor, emotion, folds, cfg, seed+1000*i)
        hard = evaluate_once(hard_task_release(P), actor, emotion, seed+3000*i, "task-label-onehot")
        candidates.append({
            "kind":"hard-task","config":cfg,
            "release_eligible":bool(hard["v22_gate"]["release_eligible"]),
            "clip_effective_auc_upper":float(hard["v22_gate"]["observed"]["clip_effective_auc_upper"]),
            "repeated_effective_auc":float(hard["repeated_effective_auc"]),
            "task_f1_lower_ci":float(hard["emotion_macro_f1_ci95"][0]),
            "task_f1":float(hard["emotion_macro_f1"]),
        })
        stochastic=[]
        for j,alpha in enumerate([0.10,0.20,0.30,0.40,0.50,0.60]):
            ev = evaluate_stochastic(P, actor, emotion, seed+5000*i+100*j, alpha)
            stochastic.append(ev)
            s=ev["conservative_summary"]
            candidates.append({
                "kind":"privacy-blanket","config":cfg,"alpha":alpha,
                "release_eligible":bool(ev["release_eligible_all_replicates"]),
                "clip_effective_auc_upper":float(s["max_clip_effective_auc_upper"]),
                "repeated_effective_auc":float(s["max_repeated_effective_auc"]),
                "task_f1_lower_ci":float(s["min_task_f1_lower_ci"]),
                "task_f1":float(np.mean([r["emotion_macro_f1"] for r in ev["replicates"]])),
            })
        sweep.append({"config":cfg,"folds":fold_rows,"hard_task_release":hard,"stochastic":stochastic})

    def rank(c):
        privacy=max(c["clip_effective_auc_upper"],c["repeated_effective_auc"])
        return (privacy,-c["task_f1_lower_ci"],-c["task_f1"])
    eligible=[c for c in candidates if c["release_eligible"]]
    result={
        "dataset":"CREMA-D DFA cohort",
        "model":"TAPF-MIN v2.3 prospective identity-invariant experiment",
        "clips":len(files),"actors":int(len(np.unique(actor))),
        "goal":"minimize identity advantage |AUC-0.5|; values below 0.5 are reflected because an attacker can invert them",
        "frozen_gate":{"max_effective_auc":0.55,"min_task_f1_lower_ci":0.20},
        "methods":["alternating identity adversary","uniform identity confusion","within-emotion actor-center collapse","stochastic latent bottleneck","task-only release","privacy blanket","repeated-release attack"],
        "sweep":sweep,"candidate_table":candidates,
        "best_overall":min(candidates,key=rank),
        "best_eligible":min(eligible,key=rank) if eligible else None,
        "eligible_operating_points":eligible,
        "integrity":{"threshold_relaxation":False,"actor_disjoint":True,"chance_centered_auc":True,"fail_closed":True},
        "claim_boundary":"Research benchmark only; not formal anonymity, differential privacy, clinical validation, or regulatory evidence.",
    }
    Path("results").mkdir(exist_ok=True)
    out=Path("results/cremad_v23_identity_invariant.json")
    out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--folds",type=int,default=5); ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args(); run(args.root,args.folds,args.seed)
