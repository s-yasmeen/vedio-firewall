from datetime import datetime, timedelta, timezone
from pathlib import Path

from tapf.consent import ConsentRecord, ConsentState, consent_allows_release
from tapf.provenance import EvidenceProvenance
from tapf.release_ledger import ReleaseEvent, ReleaseLedger


def _consent(**overrides):
    now = datetime.now(timezone.utc)
    data = dict(
        consent_id="consent-001",
        session_id="session-001",
        patient_id_hash="patient-hash",
        task="movement",
        purpose="facial movement assessment",
        recipient_id="clinic-a",
        representation="motion-features",
        nonce="nonce-001",
        state=ConsentState.ACTIVE,
        issued_at=(now - timedelta(minutes=1)).isoformat(),
        expires_at=(now + timedelta(minutes=5)).isoformat(),
        revoked_at=None,
    )
    data.update(overrides)
    return ConsentRecord(**data)


def test_active_consent_allows_release():
    out = consent_allows_release(_consent())
    assert out["allowed"] is True
    assert out["effective_state"] == "ACTIVE"
    assert len(out["consent_digest"]) == 64


def test_revoked_consent_blocks_release():
    now = datetime.now(timezone.utc)
    out = consent_allows_release(_consent(state=ConsentState.REVOKED, revoked_at=now.isoformat()))
    assert out["allowed"] is False
    assert out["effective_state"] == "REVOKED"


def test_expired_consent_blocks_release():
    now = datetime.now(timezone.utc)
    out = consent_allows_release(_consent(
        issued_at=(now - timedelta(hours=2)).isoformat(),
        expires_at=(now - timedelta(hours=1)).isoformat(),
    ))
    assert out["allowed"] is False
    assert out["effective_state"] == "EXPIRED"


def test_consent_requires_nonce_for_replay_binding():
    out = consent_allows_release(_consent(nonce=""))
    assert out["allowed"] is False
    assert out["checks"]["nonce_present"] is False


def test_release_ledger_persists_release_metadata(tmp_path: Path):
    ledger = ReleaseLedger(tmp_path / "ledger.sqlite")
    event = ReleaseEvent(
        event_id="event-001", session_id="session-001", patient_id_hash="patient-hash",
        task="movement", purpose="movement assessment", recipient_id="clinic-a",
        representation="motion-features", decision="RELEASE", privacy_effective_auc=0.53,
        repeated_effective_auc=0.54, task_f1_lower_ci=0.25,
        contract_digest="c"*64, consent_digest="d"*64, evidence_digest="e"*64,
        policy_version="v2.2-frozen", model_version="v2.3", created_at=datetime.now(timezone.utc).isoformat(),
    )
    ledger.append(event)
    assert ledger.release_count(patient_id_hash="patient-hash", session_id="session-001") == 1
    rows = ledger.recent(patient_id_hash="patient-hash")
    assert rows[0]["representation"] == "motion-features"
    assert "raw" not in str(rows[0]).lower()


def test_provenance_digest_is_stable_and_complete():
    p = EvidenceProvenance(
        code_commit="abc123", model_version="v2.3", policy_version="v2.2-frozen",
        dataset_name="CREMA-D", dataset_version="official", preprocessing_version="motion-v1",
        attacker_version="ensemble-v1", evaluator_id="validator", seed=42,
        protocol_id="cremad-v23-identity-invariant",
    )
    block = p.as_attestation_block()
    assert len(block["provenance_sha256"]) == 64
    assert block["seed"] == 42
    assert block["model_version"] == "v2.3"
