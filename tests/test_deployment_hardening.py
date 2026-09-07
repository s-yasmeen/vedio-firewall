import numpy as np
import pytest

from tapf.calibration import bootstrap_interval, wilson_interval
from tapf.cancelable import protect_embedding, revocation_check
from tapf.deployment import BoundedEvidence, DeploymentPolicy, allowed_representations, select_minimum_release
from tapf.ensemble import aggregate_worst_case
from tapf.signing import sign_attestation, verify_attestation
from tapf.transforms import FaceLocalizationError, apply_transform_strict
import tapf.edge_pipeline as edge_pipeline


def _e(alpha, privacy_u, utility_l, temporal_u, latency=50.0, evaluator='test-evaluator-v1'):
    return BoundedEvidence(
        alpha=alpha,
        identity_risk=max(0.0, privacy_u - 0.05),
        identity_risk_upper=privacy_u,
        task_utility=min(1.0, utility_l + 0.05),
        task_utility_lower=utility_l,
        temporal_risk=max(0.0, temporal_u - 0.05),
        temporal_risk_upper=temporal_u,
        latency_ms=latency,
        evaluator_id=evaluator,
        sample_count=100,
    )


def test_bounded_gate_selects_minimum_valid_alpha():
    policy = DeploymentPolicy()
    rows = [_e(0.0, .8, .9, .7), _e(.5, .2, .8, .2), _e(1.0, .1, .76, .1)]
    out = select_minimum_release('facial-expression', 'action-units', rows, policy)
    assert out['decision'] == 'RELEASE'
    assert out['selected_alpha'] == .5


def test_bounded_gate_blocks_bad_latency_and_unauthorized_representation():
    policy = DeploymentPolicy(latency_ms_threshold=100)
    out = select_minimum_release('authentication', 'protected-video', [_e(.5, .1, .9, .1)], policy)
    assert out['decision'] == 'BLOCK'
    assert out['reason'] == 'representation_not_authorized_for_task'
    out = select_minimum_release('facial-expression', 'action-units', [_e(.5, .1, .9, .1, 200)], policy)
    assert out['decision'] == 'BLOCK'


def test_representation_policy_minimizes_face_release():
    reps = allowed_representations('facial-expression')
    assert reps[0] == 'action-units'
    assert 'protected-video' in reps
    assert allowed_representations('authentication') == ('cancelable-template',)


def test_worst_case_ensemble_uses_strongest_attacker():
    weak = [_e(.5, .20, .82, .20, evaluator='hog')]
    strong = [_e(.5, .90, .80, .88, evaluator='facenet')]
    merged = aggregate_worst_case({'hog': weak, 'facenet': strong})
    assert merged[0].identity_risk_upper == .90
    assert merged[0].temporal_risk_upper == .88
    out = select_minimum_release('facial-expression', 'action-units', merged, DeploymentPolicy())
    assert out['decision'] == 'BLOCK'


def test_signed_attestation_detects_tampering():
    payload = {'release_decision': 'RELEASE', 'selected_alpha': .5}
    signed = sign_attestation(payload, 'secret')
    assert verify_attestation(signed, 'secret')
    signed['selected_alpha'] = .6
    assert not verify_attestation(signed, 'secret')


def test_calibration_helpers_produce_ordered_bounds():
    ci = bootstrap_interval([.1, .2, .3, .2, .1], resamples=200)
    assert ci.lower <= ci.mean <= ci.upper
    wi = wilson_interval(90, 100)
    assert wi.lower <= wi.mean <= wi.upper


def test_cancelable_template_key_rotation_changes_template():
    emb = np.linspace(-1, 1, 64)
    a = protect_embedding(emb, 'key-a')
    b = protect_embedding(emb, 'key-b')
    assert a.key_id != b.key_id
    assert not np.allclose(a.values, b.values)
    check = revocation_check(emb, 'key-a', 'key-b', max_cross_similarity=1.0)
    assert check['old_key_id'] != check['new_key_id']


def test_strict_transform_never_allows_raw_original():
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    with pytest.raises(FaceLocalizationError):
        apply_transform_strict('original', frame, 0.0)


def test_edge_motion_representation_contains_no_raw_frames(monkeypatch):
    monkeypatch.setattr(edge_pipeline, '_strict_face_roi', lambda frame: (10, 10, 60, 60))
    frames=[]
    for shift in (0, 2, 4):
        f=np.zeros((96,96,3),dtype=np.uint8)
        f[30:50,30+shift:50+shift]=255
        frames.append(f)
    out=edge_pipeline.derive_representation('movement', frames)
    assert out.representation == 'motion-features'
    assert out.raw_biometric_transmitted is False
    assert isinstance(out.payload, np.ndarray)
    assert out.payload.shape[1] == 4


def test_edge_expression_fails_closed_without_au_model():
    frames=[np.zeros((96,96,3),dtype=np.uint8)]
    with pytest.raises(RuntimeError, match='Action-unit extractor'):
        edge_pipeline.derive_representation('facial-expression', frames)


def test_edge_authentication_emits_cancelable_template():
    frames=[np.zeros((96,96,3),dtype=np.uint8)]
    out=edge_pipeline.derive_representation(
        'authentication', frames,
        embedding_extractor=lambda _: np.linspace(-1,1,64),
        cancelable_secret='deployment-key',
    )
    assert out.representation == 'cancelable-template'
    assert out.raw_biometric_transmitted is False
    assert 'key_id' in out.payload and len(out.payload['values']) == 64
