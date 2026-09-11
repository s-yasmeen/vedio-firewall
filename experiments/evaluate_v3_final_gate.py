"""Fail-closed evaluator for the predeclared TAPF-MIN v3 final protocol."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha256_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def evaluate(protocol_path, audit_result_path):
    protocol=json.loads(Path(protocol_path).read_text()); result=json.loads(Path(audit_result_path).read_text())
    if protocol.get('schema')!='tapf-min-final-protocol/v1': raise ValueError('wrong protocol schema')
    p=protocol['empirical_privacy']; u=protocol['utility']
    fer=result['local_fer_on_privacy_audit_cohort']; privacy=result['empirical_privacy_of_nonprivate_posterior']
    clip=privacy['clip_identity']; repeated=privacy['repeated_release_identity']
    checks={
      'fixed_model': bool(result.get('same_frozen_model_for_all_audit_actors')),
      'zero_actor_overlap': int(result.get('actor_overlap',-1))==0,
      'utility_lower_ci': float(fer['actor_cluster_macro_f1_ci95'][0]) >= float(u['min_lower_ci']),
      'clip_identity': float(clip.get('effective_auc_upper_ci95',1.0)) <= float(p['clip_identity_max_effective_auc']),
      'repeated_identity': float(repeated.get('effective_auc',1.0)) <= float(p['repeated_release_max_effective_auc']),
    }
    # Final release remains blocked until required extended attacks are supplied by a future result schema.
    ext=result.get('extended_privacy_attacks',{})
    for name in p.get('required_extensions',[]): checks['attack_'+name]=bool(ext.get(name,{}).get('pass',False))
    decision='PASS' if all(checks.values()) else 'BLOCK'
    out={'decision':decision,'checks':checks,'protocol_sha256':sha256_file(protocol_path),'result_sha256':sha256_file(audit_result_path),
         'scope':'controlled research gate; not clinical validation or anonymity proof'}
    Path('results').mkdir(exist_ok=True); Path('results/tapf_v3_final_gate.json').write_text(json.dumps(out,indent=2))
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('protocol'); ap.add_argument('audit_result'); a=ap.parse_args()
    print(json.dumps(evaluate(a.protocol,a.audit_result),indent=2))
