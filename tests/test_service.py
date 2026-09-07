from datetime import datetime, timezone, timedelta
import os
from fastapi.testclient import TestClient
from service.app import app

client = TestClient(app)


def test_healthz_is_fail_closed_and_no_raw_ingestion():
    r = client.get('/healthz')
    assert r.status_code == 200
    data = r.json()
    assert data['status'] == 'ok'
    assert data['fail_closed'] is True
    assert data['raw_biometric_ingestion'] is False
    assert data['bounded_release_api'] is True


def test_release_selects_minimum_valid_alpha():
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        'task': 'facial-expression',
        'representation': 'protected-video',
        'measured_at': now,
        'operating_points': [
            {'alpha': 0.0, 'identity_risk': 0.8, 'task_utility': 0.9, 'temporal_risk': 0.7},
            {'alpha': 0.5, 'identity_risk': 0.2, 'task_utility': 0.8, 'temporal_risk': 0.2},
            {'alpha': 1.0, 'identity_risk': 0.1, 'task_utility': 0.76, 'temporal_risk': 0.1},
        ],
    }
    r = client.post('/v1/release/evaluate', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data['decision'] == 'RELEASE'
    assert data['selected_alpha'] == 0.5
    assert data['raw_biometric_ingestion'] is False


def test_stale_evidence_blocks():
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    payload = {
        'task': 'facial-expression',
        'representation': 'protected-video',
        'measured_at': stale,
        'max_evidence_age_seconds': 60,
        'operating_points': [
            {'alpha': 0.5, 'identity_risk': 0.1, 'task_utility': 0.9, 'temporal_risk': 0.1}
        ],
    }
    r = client.post('/v1/release/evaluate', json=payload)
    assert r.status_code == 200
    assert r.json()['decision'] == 'BLOCK'
    assert r.json()['reason'] == 'stale_evidence'


def _bounded_payload():
    return {
        'task': 'facial-expression',
        'representation': 'action-units',
        'measured_at': datetime.now(timezone.utc).isoformat(),
        'operating_points': [
            {
                'alpha': 0.25,
                'identity_risk': 0.30, 'identity_risk_upper': 0.35,
                'task_utility': 0.90, 'task_utility_lower': 0.85,
                'temporal_risk': 0.25, 'temporal_risk_upper': 0.32,
                'latency_ms': 40, 'evaluator_id': 'ensemble-v1', 'sample_count': 200,
            },
            {
                'alpha': 0.50,
                'identity_risk': 0.15, 'identity_risk_upper': 0.20,
                'task_utility': 0.82, 'task_utility_lower': 0.78,
                'temporal_risk': 0.15, 'temporal_risk_upper': 0.20,
                'latency_ms': 55, 'evaluator_id': 'ensemble-v1', 'sample_count': 200,
            },
        ],
    }


def test_v2_uses_conservative_bounds_and_latency():
    r = client.post('/v2/release/evaluate', json=_bounded_payload())
    assert r.status_code == 200
    data = r.json()
    assert data['decision'] == 'RELEASE'
    assert data['selected_alpha'] == 0.5
    assert data['evaluation']['identity_risk_upper'] == 0.20
    assert data['evaluation']['task_utility_lower'] == 0.78
    assert data['evaluation']['latency_pass'] is True


def test_v2_rejects_representation_not_allowed_for_authentication():
    payload = _bounded_payload()
    payload['task'] = 'authentication'
    payload['representation'] = 'protected-video'
    r = client.post('/v2/release/evaluate', json=payload)
    assert r.status_code == 200
    assert r.json()['decision'] == 'BLOCK'
    assert r.json()['reason'] == 'representation_not_authorized_for_task'


def test_representation_discovery_prefers_non_visual_expression_signal():
    r = client.get('/v1/tasks/facial-expression/representations')
    assert r.status_code == 200
    reps = r.json()['allowed_representations']
    assert reps[0] == 'action-units'


def test_v2_attestation_can_be_signed(monkeypatch):
    monkeypatch.setenv('TAPF_ATTESTATION_SECRET', 'unit-test-secret')
    r = client.post('/v2/release/evaluate', json=_bounded_payload())
    assert r.status_code == 200
    signature = r.json()['attestation'].get('signature')
    assert signature and signature['algorithm'] == 'HMAC-SHA256'
