"""Consolidated final-validation benchmark for TAPF-MIN on the bounded CREMA-D subset.

This benchmark freezes a stronger evaluation protocol for competition/publication evidence:
- actor-disjoint task folds,
- a tighter adversarial-strength sweep,
- multiple minimum-disclosure releases,
- four independent post-hoc identity attackers including an MLP,
- repeated-release aggregation attack,
- bootstrap confidence intervals,
- conservative fail-closed gate.

Scientific scope: this remains a controlled CREMA-D emotion benchmark on the validated
147-D motion descriptor. It is not clinical validation and does not prove anonymity.
"""
from __future__ import annotations

from pathlib import Path
import argparse, json, re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from tapf.minimum_representation import read_motion_clip
from experiments.run_cremad_v21_adversarial import (
    EMOTIONS, crossfit, quantize_probabilities,
)
from experiments.run_cremad_privacy_utility_spectrum import (
    bootstrap_metric, macro_ovr_auc, identity_split,
)

NAME_RE = re.compile(r"^(\d{4})_DFA_(ANG|DIS|FEA|HAP|NEU|SAD)_XX\.flv$", re.I)
MIN_REAL_VIDEO_BYTES = 1000


def attacker_models(seed: int):
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed)),
        "rbf_svm": make_pipeline(StandardScaler(), SVC(
            C=3.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        "random_forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=1,
            random_state=seed, n_jobs=-1),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
            max_iter=1200, early_stopping=False, random_state=seed)),
    }


def _attack(Z, actor, seed: int, bootstrap_n: int = 300):
    Z = np.asarray(Z, np.float32)
    if Z.ndim == 1: Z = Z[:, None]
    tr, te = identity_split(actor, seed)
    rows=[]; worst=None
    for i,(name,model) in enumerate(attacker_models(seed).items()):
        model.fit(Z[tr], actor[tr])
        proba=model.predict_proba(Z[te]); classes=model.classes_
        auc,lo,hi=bootstrap_metric(
            actor[te],proba,lambda y,p:macro_ovr_auc(y,p,classes),
            seed=seed+100+i,n=bootstrap_n)
        row={"attacker":name,"auc":float(auc),"ci95":[float(lo),float(hi)]}
        rows.append(row)
        if worst is None or row["auc"]>worst["auc"]: worst=row
    return {"attackers":rows,"worst_case_attacker":worst["attacker"],
            "identity_auc":worst["auc"],"identity_auc_ci95":worst["ci95"],
            "identity_risk_0to1":float(min(1.0,2*abs(worst["auc"]-0.5)))}


def repeated_release_attack(Z, actor, seed: int):
    """Aggregate repeated releases before identity inference.

    Each actor has six DFA clips in the bounded subset. We form two disjoint 3-clip
    aggregates per actor: one identity-enrolment aggregate and one attack/test aggregate.
    This tests whether repeated disclosure makes identity easier to recover.
    """
    rng=np.random.default_rng(seed)
    ztr=[]; ytr=[]; zte=[]; yte=[]
    for a in sorted(np.unique(actor)):
        ids=np.flatnonzero(actor==a).copy(); rng.shuffle(ids)
        if len(ids)<4: continue
        cut=len(ids)//2
        left,right=ids[:cut],ids[cut:]
        ztr.append(np.mean(Z[left],axis=0)); ytr.append(a)
        zte.append(np.mean(Z[right],axis=0)); yte.append(a)
    ztr=np.asarray(ztr,np.float32); zte=np.asarray(zte,np.float32)
    ytr=np.asarray(ytr); yte=np.asarray(yte)
    rows=[]; worst=None
    for name,model in attacker_models(seed+50).items():
        model.fit(ztr,ytr)
        proba=model.predict_proba(zte); classes=model.classes_
        auc=macro_ovr_auc(yte,proba,classes)
        row={"attacker":name,"auc":float(auc)}; rows.append(row)
        if worst is None or row["auc"]>worst["auc"]: worst=row
    return {"aggregation":"mean of disjoint repeated releases per actor",
            "groups_per_actor":2,"clips_per_group_approx":3,
            "attackers":rows,"worst_case_attacker":worst["attacker"],
            "identity_auc":worst["auc"],
            "identity_risk_0to1":float(min(1.0,2*abs(worst["auc"]-0.5)))}


def evaluate_release(Z, actor, emotion, seed: int, label: str):
    pred=EMOTIONS[np.argmax(Z,axis=1)]
    f1,lo,hi=bootstrap_metric(
        emotion,pred,lambda y,p:f1_score(y,p,average="macro",zero_division=0),
        seed=seed,n=400)
    privacy=_attack(Z,actor,seed+500)
    temporal=repeated_release_attack(Z,actor,seed+900)
    return {
        "release":label,"dimension":int(Z.shape[1]),
        "emotion_macro_f1":float(f1),"emotion_macro_f1_ci95":[float(lo),float(hi)],
        "clip_identity":privacy,"repeated_release_identity":temporal,
        "privacy_gate_point_auc_le_0_60":bool(privacy["identity_auc"]<=0.60),
        "privacy_gate_upper_ci_le_0_60":bool(privacy["identity_auc_ci95"][1]<=0.60),
        "temporal_gate_auc_le_0_60":bool(temporal["identity_auc"]<=0.60),
        "utility_gate_lower_ci_ge_0_20":bool(lo>=0.20),
    }


def run(root, folds=5, seed=42):
    root=Path(root); files=[]
    for p in sorted(root.glob("*.flv")):
        m=NAME_RE.match(p.name)
        if m and p.stat().st_size>MIN_REAL_VIDEO_BYTES:
            files.append((p,m.group(1),m.group(2).upper()))
    actor=np.asarray([a for _,a,_ in files]); emotion=np.asarray([e for _,_,e in files])
    if len(files)<120 or len(np.unique(actor))<8:
        raise RuntimeError(f"Insufficient hydrated CREMA-D subset: {len(files)} clips/{len(np.unique(actor))} actors")
    X=np.asarray([read_motion_clip(str(p)) for p,_,_ in files],np.float32)
    folds=max(2,min(int(folds),len(np.unique(actor))))

    strengths=[0.0,0.05,0.10,0.15,0.25,0.35,0.50]
    rows=[]
    for i,s in enumerate(strengths):
        P,fold_rows=crossfit(X,actor,emotion,folds,s,seed+1000*i)
        reps=[
            ("posterior-6d",P),
            ("posterior-q8",quantize_probabilities(P,8)),
            ("posterior-q4",quantize_probabilities(P,4)),
            ("posterior-q2",quantize_probabilities(P,2)),
            ("task-label-onehot",np.eye(len(EMOTIONS),dtype=np.float32)[np.argmax(P,axis=1)]),
        ]
        evals=[]
        for j,(name,Z) in enumerate(reps):
            evals.append(evaluate_release(Z,actor,emotion,seed+10000*i+100*j,name))
        rows.append({"adversarial_strength":s,"folds":fold_rows,"releases":evals})

    candidates=[]
    for r in rows:
        for e in r["releases"]:
            c=dict(e); c["adversarial_strength"]=r["adversarial_strength"]
            c["conservative_release_eligible"]=bool(
                c["privacy_gate_upper_ci_le_0_60"] and c["temporal_gate_auc_le_0_60"] and c["utility_gate_lower_ci_ge_0_20"])
            candidates.append(c)

    eligible=[c for c in candidates if c["conservative_release_eligible"]]
    best_privacy=min(candidates,key=lambda c:abs(c["clip_identity"]["identity_auc"]-0.5))
    best_utility=max(candidates,key=lambda c:c["emotion_macro_f1"])
    best_joint=min(candidates,key=lambda c:(
        max(0,c["clip_identity"]["identity_auc"]-0.60)+max(0,c["repeated_release_identity"]["identity_auc"]-0.60),
        -c["emotion_macro_f1"]))

    result={
        "dataset":"CREMA-D bounded DFA subset","clips":len(files),"actors":int(len(np.unique(actor))),
        "model":"TAPF-MIN v2.1 consolidated feature-level validation",
        "scientific_scope":"Controlled emotion-utility benchmark on 147-D motion input; not raw-video end-to-end clinical validation and not formal anonymity.",
        "protocol":{
            "utility":f"{folds}-fold actor-disjoint out-of-fold emotion prediction",
            "clip_privacy":"closed-set identity inference using Logistic, RBF-SVM, RandomForest, and independent MLP; worst AUC reported with bootstrap CI",
            "repeated_release_privacy":"two disjoint per-actor aggregates attacked after averaging repeated releases",
            "release_gate":"upper clip-identity AUC CI <= 0.60 AND repeated-release AUC <= 0.60 AND lower emotion F1 CI >= 0.20",
        },
        "targets":{"chance_identity_auc":0.5,"privacy_auc":0.60,"utility_lower_f1":0.20},
        "sweep":rows,"eligible_operating_points":eligible,
        "best_privacy_point":best_privacy,"best_utility_point":best_utility,"best_joint_point":best_joint,
        "edge_checks":{"actor_disjoint_utility":all(fr["actor_overlap"]==0 for r in rows for fr in r["folds"]),
                       "four_independent_posthoc_attackers":True,"repeated_release_attack":True,
                       "fail_closed":True,"raw_face_released":False},
    }
    Path("results").mkdir(exist_ok=True)
    out=Path("results/cremad_final_validation.json"); out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2)); return result


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--folds",type=int,default=5)
    args=ap.parse_args(); run(args.root,args.folds)
