"""Two-sided task authorization for TAPF-MIN.

The doctor/clinical system declares the authorized purpose and recipient. The
patient-side firewall validates that request before evaluating any biometric release.
This is an architectural authorization control; it does not replace empirical
privacy/utility validation and does not constitute regulatory consent management.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json

from tapf.deployment import allowed_representations


@dataclass(frozen=True)
class TaskAuthorizationContract:
    session_id: str
    task: str
    purpose: str
    recipient_id: str
    requested_representation: str
    patient_authorized: bool
    issued_at: str
    expires_at: str | None = None

    def canonical_payload(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        payload = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


def _parse_time(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def validate_task_contract(contract: TaskAuthorizationContract,
                           *,
                           expected_recipient_id: str | None = None,
                           now: datetime | None = None) -> dict:
    """Validate doctor-side request and patient-side authorization fail-closed."""
    task = contract.task.strip().lower()
    representation = contract.requested_representation.strip().lower()
    allowed = allowed_representations(task)

    checks = {
        "session_present": bool(contract.session_id.strip()),
        "task_present": bool(task),
        "purpose_present": bool(contract.purpose.strip()),
        "recipient_present": bool(contract.recipient_id.strip()),
        "patient_authorized": bool(contract.patient_authorized),
        "representation_authorized_for_task": representation in allowed,
    }

    if expected_recipient_id is not None:
        checks["recipient_bound"] = contract.recipient_id == expected_recipient_id

    try:
        issued = _parse_time(contract.issued_at)
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        checks["issued_at_valid"] = issued <= current
    except Exception:
        issued = None
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        checks["issued_at_valid"] = False

    if contract.expires_at:
        try:
            expiry = _parse_time(contract.expires_at)
            checks["not_expired"] = current <= expiry
            checks["expiry_after_issue"] = issued is not None and expiry >= issued
        except Exception:
            checks["not_expired"] = False
            checks["expiry_after_issue"] = False

    authorized = all(checks.values())
    return {
        "authorized": authorized,
        "checks": checks,
        "task": task,
        "purpose": contract.purpose,
        "recipient_id": contract.recipient_id,
        "requested_representation": representation,
        "allowed_representations": allowed,
        "session_id": contract.session_id,
        "contract_digest": contract.digest(),
    }
