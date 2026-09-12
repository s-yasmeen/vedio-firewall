from datetime import datetime, timezone, timedelta
import os
from fastapi.testclient import TestClient
from service.app import app

client = TestClient(app)


def _auth_headers():
    key=os.getenv('TAPF_API_KEY')
    return {'Authorization':f'Bearer {key}'} if key else {}


def test_healthz_is_fail_closed_and_no_raw_ingestion():
    r=client.get('/healthz'); assert r.status_code==200; data=r.json()
    assert data['status']=='ok' and data['fail_closed'] is True
    assert data['raw_biometric_ingestion'] is False and data['bounded_release_api'] is True
    assert data['composition_guard'] is True and data['v22_chance_centered_gate'] is True
    assert data['two_sided_task_authorization'] is True
    assert data['preferred_release_endpoint']=='/v23/authorized-release/evaluate'


def test_readyz_requires_security_configuration(monkeypatch):
    monkeypatch.delenv('TAPF_ATTESTATION_SECRET',raising=False); monkeypatch.delenv('TAPF_API_KEY',raising=False)
    assert client.get('/readyz').status_code==503
    monkeypatch.setenv('TAPF_ATTESTATION_SECRET','s'); monkeypatch.setenv('TAPF_API_KEY','k')
    r=client.get('/readyz'); assert r.status_code==200 and r.json()['ready'] is True


def test_release_selects_minimum_valid_alpha_with_explicit_legacy_policy():
    payload={'task':'facial-expression','representation':'protected-video','measured_at':datetime.now(timezone.utc).isoformat(),
      'privacy_threshold':0.25,'utility_threshold':0.75,'temporal_threshold':0.30,
      'operating_points':[{'alpha':0.0,'identity_risk':0.8,'task_utility':0.9,'temporal_risk':0.7},{'alpha':0.5,'identity_risk':0.2,'task_utility':0.8,'temporal_risk':0.2},{'alpha':1.0,'identity_risk':0.1,'task_utility':0.76,'temporal_risk':0.1}]}
    r=client.post('/v1/release/evaluate',json=payload); assert r.status_code==200
    assert r.json()['decision']=='RELEASE' and r.json()['selected_alpha']==0.5


def test_stale_evidence_blocks():
    payload={'task':'facial-expression','representation':'protected-video','measured_at':(datetime.now(timezone.utc)-timedelta(hours=2)).isoformat(),'max_evidence_age_seconds':60,
      'operating_points':[{'alpha':0.5,'identity_risk':0.1,'task_utility':0.9,'temporal_risk':0.1}]}
    r=client.post('/v1/release/evaluate',json=payload); assert r.json()['decision']=='BLOCK' and r.json()['reason']=='stale_evidence'


def _bounded_payload():
    return {'request_id':'unit-test-request-001','task':'facial-expression','representation':'action-units','measured_at':datetime.now(timezone.utc).isoformat(),
      'privacy_upper_threshold':0.25,'utility_lower_threshold':0.75,'temporal_upper_threshold':0.30,'composition_upper_threshold':0.30,
      'operating_points':[{'alpha':0.25,'identity_risk':0.30,'identity_risk_upper':0.35,'task_utility':0.90,'task_utility_lower':0.85,'temporal_risk':0.25,'temporal_risk_upper':0.32,'latency_ms':40,'evaluator_id':'ensemble-v1','sample_count':200},
      {'alpha':0.50,'identity_risk':0.15,'identity_risk_upper':0.20,'task_utility':0.82,'task_utility_lower':0.78,'temporal_risk':0.15,'temporal_risk_upper':0.20,'latency_ms':55,'evaluator_id':'ensemble-v1','sample_count':200}]}


def test_v2_uses_conservative_bounds_latency_and_traceability():
    r=client.post('/v2/release/evaluate',json=_bounded_payload(),headers=_auth_headers()); assert r.status_code==200; data=r.json()
    assert data['decision']=='RELEASE' and data['selected_alpha']==0.5
    assert data['evaluation']['identity_risk_upper']==0.20 and data['evaluation']['task_utility_lower']==0.78
    assert data['evaluation']['latency_pass'] is True and len(data['evidence_sha256'])==64


def test_v2_repeated_release_requires_composition_evidence():
    payload=_bounded_payload(); payload['prior_release_count']=1
    data=client.post('/v2/release/evaluate',json=payload,headers=_auth_headers()).json()
    assert data['decision']=='BLOCK' and data['reason']=='composition_evidence_required'


def test_v2_repeated_release_blocks_unsafe_composition():
    payload=_bounded_payload(); payload.update({'prior_release_count':1,'composition_risk_upper':0.42,'composition_evaluator_id':'repeat-attack-v1'})
    data=client.post('/v2/release/evaluate',json=payload,headers=_auth_headers()).json()
    assert data['decision']=='BLOCK' and data['reason']=='composition_risk_exceeds_threshold'


def test_v2_repeated_release_can_pass_with_bounded_composition_evidence():
    payload=_bounded_payload(); payload.update({'prior_release_count':1,'composition_risk_upper':0.20,'composition_evaluator_id':'repeat-attack-v1'})
    data=client.post('/v2/release/evaluate',json=payload,headers=_auth_headers()).json()
    assert data['decision']=='RELEASE' and data['attestation']['composition']['guard_pass'] is True


def _v22_payload(**overrides):
    point={'alpha':0.5,'clip_auc_ci95_low':0.48,'clip_auc_ci95_high':0.54,'repeated_release_auc':0.53,
           'task_f1_ci95_low':0.24,'task_f1_ci95_high':0.31,'attacker_aucs':[0.51,0.52,0.49,0.53],
           'latency_ms':80,'evaluator_id':'v22-ensemble','sample_count':200}
    point.update(overrides)
    return {'request_id':'v22-unit-request','task':'emotion','representation':'expression-embedding',
            'measured_at':datetime.now(timezone.utc).isoformat(),'operating_points':[point]}


def test_v22_strict_gate_releases_only_when_all_constraints_pass():
    data=client.post('/v22/release/evaluate',json=_v22_payload(),headers=_auth_headers()).json()
    assert data['decision']=='RELEASE'
    assert data['attestation']['thresholds']['max_effective_auc']==0.55


def test_v22_blocks_repeated_release_leakage():
    data=client.post('/v22/release/evaluate',json=_v22_payload(repeated_release_auc=0.64),headers=_auth_headers()).json()
    assert data['decision']=='BLOCK'


def _v23_payload(*, patient_authorized=True, representation='motion-features', repeated_release_auc=0.53):
    now=datetime.now(timezone.utc)
    return {
      'request_id':'v23-unit-request',
      'contract':{
        'session_id':'session-001','task':'movement','purpose':'assess facial movement',
        'recipient_id':'clinic-a','requested_representation':representation,
        'patient_authorized':patient_authorized,'issued_at':now.isoformat(),
        'expires_at':(now+timedelta(minutes=15)).isoformat(),
      },
      'measured_at':now.isoformat(),
      'operating_points':[{
        'alpha':0.5,'clip_auc_ci95_low':0.48,'clip_auc_ci95_high':0.54,
        'repeated_release_auc':repeated_release_auc,
        'task_f1_ci95_low':0.24,'task_f1_ci95_high':0.31,
        'attacker_aucs':[0.51,0.52,0.49,0.53],'latency_ms':80,
        'evaluator_id':'v23-ensemble','sample_count':200,
      }],
    }


def _v23_headers(recipient='clinic-a'):
    headers=_auth_headers(); headers['X-TAPF-Recipient-ID']=recipient; return headers


def test_v23_two_sided_authorized_request_can_release():
    r=client.post('/v23/authorized-release/evaluate',json=_v23_payload(),headers=_v23_headers())
    assert r.status_code==200; data=r.json()
    assert data['decision']=='RELEASE' and data['task_authorization_pass'] is True
    assert data['two_sided_task_authorization'] is True
    assert data['attestation']['purpose']=='assess facial movement'
    assert data['attestation']['recipient_id']=='clinic-a'
    assert data['attestation']['patient_authorized'] is True
    assert len(data['contract_sha256'])==64 and len(data['evidence_sha256'])==64


def test_v23_patient_denial_blocks_good_evidence():
    data=client.post('/v23/authorized-release/evaluate',json=_v23_payload(patient_authorized=False),headers=_v23_headers()).json()
    assert data['decision']=='BLOCK' and data['reason']=='task_authorization_contract_failed'
    assert data['attestation']['authorization_checks']['patient_authorized'] is False


def test_v23_recipient_binding_mismatch_blocks():
    data=client.post('/v23/authorized-release/evaluate',json=_v23_payload(),headers=_v23_headers('clinic-b')).json()
    assert data['decision']=='BLOCK' and data['reason']=='task_authorization_contract_failed'
    assert data['attestation']['authorization_checks']['recipient_bound'] is False


def test_v23_authorization_cannot_override_repeated_release_failure():
    data=client.post('/v23/authorized-release/evaluate',json=_v23_payload(repeated_release_auc=0.64),headers=_v23_headers()).json()
    assert data['decision']=='BLOCK' and data['task_authorization_pass'] is True


def test_v2_rejects_representation_not_allowed_for_authentication():
    payload=_bounded_payload(); payload['task']='authentication'; payload['representation']='protected-video'
    data=client.post('/v2/release/evaluate',json=payload,headers=_auth_headers()).json()
    assert data['decision']=='BLOCK' and data['reason']=='representation_not_authorized_for_task'


def test_v2_requires_auth_when_configured(monkeypatch):
    monkeypatch.setenv('TAPF_API_KEY','required-key')
    assert client.post('/v2/release/evaluate',json=_bounded_payload()).status_code==401
    assert client.post('/v2/release/evaluate',json=_bounded_payload(),headers={'Authorization':'Bearer required-key'}).status_code==200


def test_representation_discovery_prefers_non_visual_expression_signal():
    r=client.get('/v1/tasks/facial-expression/representations'); assert r.status_code==200
    assert r.json()['allowed_representations'][0]=='action-units'


def test_v2_attestation_can_be_signed(monkeypatch):
    monkeypatch.setenv('TAPF_ATTESTATION_SECRET','unit-test-secret')
    r=client.post('/v2/release/evaluate',json=_bounded_payload(),headers=_auth_headers()); assert r.status_code==200
    signature=r.json()['attestation'].get('signature'); assert signature and signature['algorithm']=='HMAC-SHA256'
