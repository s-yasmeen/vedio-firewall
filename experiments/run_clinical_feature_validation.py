"""Dataset-agnostic clinical feature validation harness for TAPF-MIN v2.2.

This script is intentionally data-source neutral. It accepts a CSV exported from an
approved clinical collaboration after local/on-premise feature extraction. Raw patient
video is not required by this harness.

Required columns:
- patient_id: stable study-local pseudonymous patient identifier
- task_label: clinically meaningful task/condition label supplied by the dataset owner
- one or more numeric feature columns

Optional:
- session_id: longitudinal session identifier (recommended)

Protocol:
- patient-disjoint task cross-validation,
- minimum-disclosure task-posterior releases,
- four independent post-hoc identity attackers,
- repeated-release identity attack,
- bootstrap 95% confidence intervals,
- frozen TAPF-MIN v2.2 chance-centered release gate.

Running this harness on synthetic/non-patient data must NOT be called clinical
validation. Clinical validation status requires an actual approved clinical cohort and
appropriate governance/ethics permissions.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.run_cremad_final_validation import _attack, repeated_release_attack
from experiments.run_cremad_privacy_utility_spectrum import bootstrap_metric
from tapf.privacy_gate_v22 import effective_auc, evaluate_v22_gate

RESERVED = {"patient_id", "task_label", "session_id"}


def load_csv(path: str | Path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "patient_id" not in reader.fieldnames or "task_label" not in reader.fieldnames:
            raise ValueError("CSV requires patient_id and task_label columns")
        features = [c for c in reader.fieldnames if c not in RESERVED]
        if not features:
            raise ValueError("CSV requires at least one numeric feature column")
        X=[]; patients=[]; labels=[]; sessions=[]
        for n,row in enumerate(reader, start=2):
            try:
                vec=[float(row[c]) for c in features]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric feature at CSV row {n}") from exc
            if not np.isfinite(vec).all():
                raise ValueError(f"Non-finite feature at CSV row {n}")
            pid=(row.get("patient_id") or "").strip(); lab=(row.get("task_label") or "").strip()
            if not pid or not lab:
                raise ValueError(f"Missing patient_id/task_label at CSV row {n}")
            X.append(vec); patients.append(pid); labels.append(lab); sessions.append((row.get("session_id") or str(n)).strip())
    return np.asarray(X,np.float32), np.asarray(patients), np.asarray(labels), np.asarray(sessions), features


def crossfit(X, patients, labels, folds: int, seed: int):
    classes=np.asarray(sorted(np.unique(labels)))
    if len(classes)<2: raise ValueError("At least two task labels are required")
    n_pat=len(np.unique(patients))
    if n_pat<4: raise ValueError("At least four patients are required")
    k=max(2,min(int(folds),n_pat))
    P=np.zeros((len(X),len(classes)),np.float32)
    mapping={c:i for i,c in enumerate(classes)}
    rows=[]
    for fold,(tr,te) in enumerate(GroupKFold(n_splits=k).split(X,labels,groups=patients)):
        m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,class_weight="balanced",random_state=seed+fold))
        m.fit(X[tr],labels[tr]); prob=m.predict_proba(X[te])
        for j,c in enumerate(m.classes_): P[te,mapping[c]]=prob[:,j]
        pred=classes[np.argmax(P[te],axis=1)]
        rows.append({"fold":fold,"patient_overlap":int(len(set(patients[tr])&set(patients[te]))),
                     "macro_f1":float(f1_score(labels[te],pred,average="macro",zero_division=0))})
    return P,classes,rows


def quantize(P, levels):
    q=np.rint(np.clip(P,0,1)*(levels-1))/(levels-1)
    s=q.sum(1,keepdims=True); z=s[:,0]==0
    if np.any(z): q[z]=1.0/q.shape[1]; s=q.sum(1,keepdims=True)
    return (q/s).astype(np.float32)


def hard(P): return np.eye(P.shape[1],dtype=np.float32)[np.argmax(P,axis=1)]


def temperature(P,t):
    l=np.log(np.clip(P,1e-8,1.0))/float(t); l-=l.max(1,keepdims=True)
    o=np.exp(l); o/=o.sum(1,keepdims=True); return o.astype(np.float32)


def evaluate(Z, patients, labels, classes, seed, name):
    pred=classes[np.argmax(Z,axis=1)]
    f1,lo,hi=bootstrap_metric(labels,pred,lambda y,p:f1_score(y,p,average="macro",zero_division=0),seed=seed,n=400)
    clip=_attack(Z,patients,seed+500); repeat=repeated_release_attack(Z,patients,seed+900)
    attacker=[float(r["auc"]) for r in clip["attackers"]]
    gate=evaluate_v22_gate(clip_auc_ci95=clip["identity_auc_ci95"],repeated_release_auc=repeat["identity_auc"],
                           task_f1_ci95=[lo,hi],attacker_aucs=attacker)
    return {"release":name,"task_macro_f1":float(f1),"task_macro_f1_ci95":[float(lo),float(hi)],
            "clip_identity":clip,"repeated_release_identity":repeat,"v22_gate":gate}


def run(path, folds=5, seed=42, cohort_name="clinical-cohort"):
    X,patients,labels,sessions,features=load_csv(path)
    P,classes,fold_rows=crossfit(X,patients,labels,folds,seed)
    candidates=[("posterior",P),("posterior-q8",quantize(P,8)),("posterior-q4",quantize(P,4)),
                ("posterior-q2",quantize(P,2)),("task-label-onehot",hard(P)),
                ("posterior-temp-2",temperature(P,2.0)),("posterior-temp-4",temperature(P,4.0))]
    rows=[evaluate(z,patients,labels,classes,seed+1000*i,n) for i,(n,z) in enumerate(candidates)]
    eligible=[r for r in rows if r["v22_gate"]["release_eligible"]]
    def rank(r):
        o=r["v22_gate"]["observed"]
        return (max(o["clip_effective_auc_upper"],o["repeated_release_effective_auc"]),-o["task_f1_lower_ci"])
    result={
        "cohort":cohort_name,"rows":int(len(X)),"patients":int(len(np.unique(patients))),
        "sessions":int(len(np.unique(sessions))),"task_labels":classes.tolist(),"feature_count":len(features),
        "raw_patient_video_used_by_harness":False,
        "protocol":"TAPF-MIN v2.2 frozen patient-disjoint privacy/utility/repeated-release protocol",
        "folds":fold_rows,"releases":rows,"eligible_count":len(eligible),
        "best_overall":min(rows,key=rank),"best_eligible":min(eligible,key=rank) if eligible else None,
        "integrity":{"patient_disjoint":all(r["patient_overlap"]==0 for r in fold_rows),
                     "threshold_relaxation_after_results":False,"fail_closed":True},
        "claim_boundary":"Clinical validation may be claimed only when this harness is run on an approved real clinical cohort with clinically meaningful labels and appropriate governance.",
    }
    Path("results").mkdir(exist_ok=True)
    out=Path("results/clinical_feature_validation_v22.json"); out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2)); return result


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("csv_path"); ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--cohort-name",default="clinical-cohort")
    a=ap.parse_args(); run(a.csv_path,a.folds,a.seed,a.cohort_name)
