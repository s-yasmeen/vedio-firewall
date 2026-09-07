"""Finalize TAPF-MIN release from confidence-bounded measured operating points."""
from __future__ import annotations
import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path
from tapf.deployment import BoundedEvidence, DeploymentPolicy, select_minimum_release
from tapf.signing import sign_attestation


def finalize(path, output, task, representation, privacy=.25, utility=.75, temporal=.30, latency=150.0):
    rows=json.loads(Path(path).read_text())
    evidence=[BoundedEvidence(**r) for r in rows]
    policy=DeploymentPolicy(privacy,utility,temporal,latency,True)
    decision=select_minimum_release(task,representation,evidence,policy)
    attestation={
        'schema':'tapf-min-attestation/v2',
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'task':task,'representation':representation,
        'release_decision':decision['decision'],'reason':decision.get('reason'),
        'selected_alpha':decision.get('selected_alpha'),'evaluation':decision.get('evaluation'),
        'thresholds':{'privacy_upper':privacy,'utility_lower':utility,'temporal_upper':temporal,'latency_ms':latency},
        'raw_biometric_transmitted':False,
        'source':str(path),
        'scope':'Empirical release decision under tested evaluators and threat model; not formal anonymity or clinical validation.'
    }
    secret=os.getenv('TAPF_ATTESTATION_SECRET')
    if secret: attestation=sign_attestation(attestation,secret,os.getenv('TAPF_ATTESTATION_KEY_ID','local-hmac'))
    payload={'decision':decision,'attestation':attestation,'source_rows':rows}
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2)); return payload

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('metrics'); p.add_argument('--output',default='results/final_bounded_attestation.json')
    p.add_argument('--task',default='facial-expression'); p.add_argument('--representation',default='action-units')
    p.add_argument('--privacy',type=float,default=.25); p.add_argument('--utility',type=float,default=.75)
    p.add_argument('--temporal',type=float,default=.30); p.add_argument('--latency',type=float,default=150.0)
    a=p.parse_args(); finalize(a.metrics,a.output,a.task,a.representation,a.privacy,a.utility,a.temporal,a.latency)
