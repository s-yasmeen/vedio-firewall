"""Consent lifecycle and anti-replay primitives for TAPF-MIN.

Research/demo control layer. This module is not a legal consent-management system and
is not a substitute for institutional or regulatory consent procedures.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json


class ConsentState(str, Enum):
    REQUESTED = "REQUESTED"
    AUTHORIZED = "AUTHORIZED"
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class ConsentRecord:
    consent_id: str
    session_id: str
    patient_id_hash: str
    task: str
    purpose: str
    recipient_id: str
    representation: str
    nonce: str
    state: ConsentState
    issued_at: str
    expires_at: str
    revoked_at: str | None = None

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return sha256(payload.encode("utf-8")).hexdigest()


def _parse(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def effective_state(record: ConsentRecord, *, now: datetime | None = None) -> ConsentState:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if record.revoked_at or record.state == ConsentState.REVOKED:
        return ConsentState.REVOKED
    try:
        if current > _parse(record.expires_at):
            return ConsentState.EXPIRED
    except Exception:
        return ConsentState.EXPIRED
    return record.state


def consent_allows_release(record: ConsentRecord, *, now: datetime | None = None) -> dict:
    state = effective_state(record, now=now)
    checks = {
        "consent_id_present": bool(record.consent_id.strip()),
        "session_present": bool(record.session_id.strip()),
        "patient_binding_present": bool(record.patient_id_hash.strip()),
        "task_present": bool(record.task.strip()),
        "purpose_present": bool(record.purpose.strip()),
        "recipient_present": bool(record.recipient_id.strip()),
        "representation_present": bool(record.representation.strip()),
        "nonce_present": bool(record.nonce.strip()),
        "state_active": state in (ConsentState.AUTHORIZED, ConsentState.ACTIVE),
    }
    try:
        issued = _parse(record.issued_at)
        expiry = _parse(record.expires_at)
        checks["time_window_valid"] = issued <= expiry
    except Exception:
        checks["time_window_valid"] = False
    return {
        "allowed": all(checks.values()),
        "effective_state": state.value,
        "checks": checks,
        "consent_digest": record.digest(),
    }
