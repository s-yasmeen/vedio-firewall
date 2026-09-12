"""Persistent exposure-ledger primitives for TAPF-MIN.

The ledger records only release metadata and evidence summaries. It must not contain raw
biometric frames. SQLite is used for a portable demo/research implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path


@dataclass(frozen=True)
class ReleaseEvent:
    event_id: str
    session_id: str
    patient_id_hash: str
    task: str
    purpose: str
    recipient_id: str
    representation: str
    decision: str
    privacy_effective_auc: float | None
    repeated_effective_auc: float | None
    task_f1_lower_ci: float | None
    contract_digest: str
    consent_digest: str
    evidence_digest: str
    policy_version: str
    model_version: str
    created_at: str


class ReleaseLedger:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._init()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init(self):
        with self._connect() as cx:
            cx.execute("""
            CREATE TABLE IF NOT EXISTS release_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                patient_id_hash TEXT NOT NULL,
                task TEXT NOT NULL,
                purpose TEXT NOT NULL,
                recipient_id TEXT NOT NULL,
                representation TEXT NOT NULL,
                decision TEXT NOT NULL,
                privacy_effective_auc REAL,
                repeated_effective_auc REAL,
                task_f1_lower_ci REAL,
                contract_digest TEXT NOT NULL,
                consent_digest TEXT NOT NULL,
                evidence_digest TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """)
            cx.execute("CREATE INDEX IF NOT EXISTS ix_release_session ON release_events(session_id)")
            cx.execute("CREATE INDEX IF NOT EXISTS ix_release_patient ON release_events(patient_id_hash)")

    def append(self, event: ReleaseEvent):
        if event.decision not in {"RELEASE", "BLOCK", "DEGRADED"}:
            raise ValueError("invalid decision")
        payload = json.dumps(asdict(event), sort_keys=True)
        with self._connect() as cx:
            cx.execute("""
            INSERT INTO release_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                event.event_id, event.session_id, event.patient_id_hash, event.task,
                event.purpose, event.recipient_id, event.representation, event.decision,
                event.privacy_effective_auc, event.repeated_effective_auc,
                event.task_f1_lower_ci, event.contract_digest, event.consent_digest,
                event.evidence_digest, event.policy_version, event.model_version,
                event.created_at, payload,
            ))

    def release_count(self, *, patient_id_hash: str, session_id: str | None = None) -> int:
        q = "SELECT COUNT(*) FROM release_events WHERE patient_id_hash=? AND decision='RELEASE'"
        args = [patient_id_hash]
        if session_id is not None:
            q += " AND session_id=?"
            args.append(session_id)
        with self._connect() as cx:
            return int(cx.execute(q, args).fetchone()[0])

    def recent(self, *, patient_id_hash: str, limit: int = 20) -> list[dict]:
        with self._connect() as cx:
            rows = cx.execute(
                "SELECT payload_json FROM release_events WHERE patient_id_hash=? ORDER BY created_at DESC LIMIT ?",
                (patient_id_hash, int(limit)),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
