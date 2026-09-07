from datetime import datetime, timezone, timedelta
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
