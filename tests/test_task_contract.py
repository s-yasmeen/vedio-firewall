from datetime import datetime, timedelta, timezone

from tapf.deployment_v22 import (
    V22BoundedEvidence,
    select_authorized_v22_release,
)
from tapf.task_contract import TaskAuthorizationContract, validate_task_contract


def _contract(**overrides):
    now = datetime.now(timezone.utc)
    data = dict(
        session_id="session-001",
        task="movement",
        purpose="assess facial movement",
        recipient_id="clinic-a",
        requested_representation="motion-features",
        patient_authorized=True,
        issued_at=(now - timedelta(minutes=1)).isoformat(),
        expires_at=(now + timedelta(minutes=15)).isoformat(),
    )
    data.update(overrides)
    return TaskAuthorizationContract(**data)


def _passing_evidence():
    return [V22BoundedEvidence(
        alpha=0.2,
        clip_auc_ci95_low=0.49,
        clip_auc_ci95_high=0.54,
        repeated_release_auc=0.53,
        task_f1_ci95_low=0.23,
        task_f1_ci95_high=0.31,
        attacker_aucs=(0.51, 0.52, 0.53),
        latency_ms=90.0,
        evaluator_id="test-evaluator",
        sample_count=100,
    )]


def test_valid_two_sided_contract_passes_authorization():
    result = validate_task_contract(_contract(), expected_recipient_id="clinic-a")
    assert result["authorized"] is True
    assert result["checks"]["recipient_bound"] is True
    assert result["checks"]["representation_authorized_for_task"] is True
    assert len(result["contract_digest"]) == 64


def test_patient_denial_blocks_even_with_good_model_evidence():
    result = select_authorized_v22_release(
        _contract(patient_authorized=False),
        _passing_evidence(),
        expected_recipient_id="clinic-a",
    )
    assert result["decision"] == "BLOCK"
    assert result["reason"] == "task_authorization_contract_failed"


def test_recipient_mismatch_blocks():
    result = select_authorized_v22_release(
        _contract(),
        _passing_evidence(),
        expected_recipient_id="clinic-b",
    )
    assert result["decision"] == "BLOCK"
    assert result["contract"]["checks"]["recipient_bound"] is False


def test_task_representation_mismatch_blocks():
    result = select_authorized_v22_release(
        _contract(requested_representation="cancelable-template"),
        _passing_evidence(),
        expected_recipient_id="clinic-a",
    )
    assert result["decision"] == "BLOCK"
    assert result["contract"]["checks"]["representation_authorized_for_task"] is False


def test_expired_contract_blocks():
    now = datetime.now(timezone.utc)
    result = select_authorized_v22_release(
        _contract(
            issued_at=(now - timedelta(hours=2)).isoformat(),
            expires_at=(now - timedelta(hours=1)).isoformat(),
        ),
        _passing_evidence(),
        expected_recipient_id="clinic-a",
    )
    assert result["decision"] == "BLOCK"
    assert result["contract"]["checks"]["not_expired"] is False


def test_authorized_contract_still_cannot_override_privacy_gate():
    failing = [V22BoundedEvidence(
        alpha=0.2,
        clip_auc_ci95_low=0.50,
        clip_auc_ci95_high=0.70,
        repeated_release_auc=0.68,
        task_f1_ci95_low=0.30,
        task_f1_ci95_high=0.40,
        attacker_aucs=(0.65,),
        latency_ms=80.0,
        evaluator_id="test-evaluator",
        sample_count=100,
    )]
    result = select_authorized_v22_release(
        _contract(),
        failing,
        expected_recipient_id="clinic-a",
    )
    assert result["decision"] == "BLOCK"
    assert result["task_authorization_pass"] is True


def test_authorized_contract_and_passing_evidence_can_release():
    result = select_authorized_v22_release(
        _contract(),
        _passing_evidence(),
        expected_recipient_id="clinic-a",
    )
    assert result["decision"] == "RELEASE"
    assert result["task_authorization_pass"] is True
