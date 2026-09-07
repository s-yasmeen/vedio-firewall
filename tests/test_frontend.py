from fastapi.testclient import TestClient
from service.app import app

client = TestClient(app)


def test_root_redirects_to_app():
    r = client.get('/', follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers['location'] == '/app/'


def test_frontend_shell_is_served():
    r = client.get('/app/')
    assert r.status_code == 200
    assert 'TAPF-MIN' in r.text
    assert 'Biometric Release Control' in r.text
    assert 'No raw video upload' in r.text


def test_pwa_assets_are_served():
    manifest = client.get('/app/manifest.webmanifest')
    assert manifest.status_code == 200
    data = manifest.json()
    assert data['name'] == 'TAPF-MIN Edge'
    assert data['display'] == 'standalone'
    assert data['start_url'] == '/app/'

    sw = client.get('/app/sw.js')
    assert sw.status_code == 200
    assert "tapf-min-ui-v1" in sw.text


def test_health_reports_ui_availability():
    r = client.get('/healthz')
    assert r.status_code == 200
    assert r.json()['ui_available'] is True
