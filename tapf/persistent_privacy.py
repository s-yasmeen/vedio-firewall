"""Persistent privacy-budget accounting for TAPF-MIN v3.

This module closes a deployment gap in the in-memory research accountant: process
restarts must not reset the cumulative privacy spend. SQLite is used as a small,
transactional reference implementation suitable for a single edge node or testbed.
Distributed deployments should provide an equivalent strongly consistent ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import sqlite3
from pathlib import Path
from typing import Optional

from tapf.formal_privacy import FormalPrivacyBudget, PrivacyBudgetExceeded, PrivacySpend


class DuplicateRelease(RuntimeError):
    """Raised when an idempotent release identifier is spent more than once."""


def _validate_spend(epsilon: float, delta: float) -> None:
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError("epsilon must be finite and > 0")
    if not math.isfinite(float(delta)) or not 0 <= float(delta) < 1:
        raise ValueError("delta must be finite and in [0,1)")


@dataclass(frozen=True)
class PersistentSpend:
    release_id: str | None
    mechanism: str
    epsilon: float
    delta: float


class SQLiteDPAccountant:
    """Restart-safe, scope-bound, atomic DP composition ledger.

    `scope_id` identifies the privacy unit/accounting scope selected by policy (for
    example one encounter or one patient-time-window). The class does not choose that
    policy; it enforces cumulative spend within the supplied scope.

    `spend_once` should be preferred at a network/service boundary because it binds a
    caller-provided idempotency key to the spend. `spend` exists for compatibility with
    the in-memory accountant interface used by the research privacy kernel.
    """

    def __init__(self, db_path: str | Path, scope_id: str, budget: FormalPrivacyBudget):
        if not scope_id or len(scope_id) > 256:
            raise ValueError("scope_id must be non-empty and <=256 characters")
        self.db_path = str(db_path)
        self.scope_id = str(scope_id)
        self.budget = budget
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS privacy_spend (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_id TEXT NOT NULL,
                    release_id TEXT,
                    mechanism TEXT NOT NULL,
                    epsilon REAL NOT NULL,
                    delta REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    UNIQUE(scope_id, release_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_privacy_spend_scope ON privacy_spend(scope_id)"
            )

    def _totals(self, conn: sqlite3.Connection | None = None) -> tuple[float, float, int]:
        owns = conn is None
        c = conn or self._connect()
        try:
            row = c.execute(
                "SELECT COALESCE(SUM(epsilon),0), COALESCE(SUM(delta),0), COUNT(*) "
                "FROM privacy_spend WHERE scope_id=?",
                (self.scope_id,),
            ).fetchone()
            return float(row[0]), float(row[1]), int(row[2])
        finally:
            if owns:
                c.close()

    @property
    def epsilon_spent(self) -> float:
        return self._totals()[0]

    @property
    def delta_spent(self) -> float:
        return self._totals()[1]

    @property
    def remaining(self) -> dict:
        eps, delta, _ = self._totals()
        return {
            "epsilon": max(0.0, self.budget.epsilon_max - eps),
            "delta": max(0.0, self.budget.delta_max - delta),
        }

    @property
    def spends(self) -> tuple[PersistentSpend, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT release_id, mechanism, epsilon, delta FROM privacy_spend "
                "WHERE scope_id=? ORDER BY id",
                (self.scope_id,),
            ).fetchall()
        return tuple(PersistentSpend(r[0], str(r[1]), float(r[2]), float(r[3])) for r in rows)

    def can_spend(self, epsilon: float, delta: float = 0.0) -> bool:
        """Advisory pre-check. `spend`/`spend_once` repeat the check atomically."""
        _validate_spend(epsilon, delta)
        eps, dlt, _ = self._totals()
        return (
            eps + float(epsilon) <= self.budget.epsilon_max + 1e-12
            and dlt + float(delta) <= self.budget.delta_max + 1e-18
        )

    def _atomic_spend(
        self,
        mechanism: str,
        epsilon: float,
        delta: float,
        release_id: Optional[str],
    ) -> PrivacySpend:
        if not mechanism:
            raise ValueError("mechanism must be non-empty")
        _validate_spend(epsilon, delta)
        if release_id is not None and (not release_id or len(release_id) > 256):
            raise ValueError("release_id must be non-empty and <=256 characters")

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if release_id is not None:
                exists = conn.execute(
                    "SELECT 1 FROM privacy_spend WHERE scope_id=? AND release_id=?",
                    (self.scope_id, release_id),
                ).fetchone()
                if exists:
                    conn.execute("ROLLBACK")
                    raise DuplicateRelease(f"release_id already spent in scope: {release_id}")

            eps_spent, delta_spent, _ = self._totals(conn)
            if (
                eps_spent + float(epsilon) > self.budget.epsilon_max + 1e-12
                or delta_spent + float(delta) > self.budget.delta_max + 1e-18
            ):
                conn.execute("ROLLBACK")
                raise PrivacyBudgetExceeded(
                    "Formal privacy budget exhausted: "
                    f"requested (eps={epsilon}, delta={delta}); "
                    f"spent (eps={eps_spent}, delta={delta_spent}); "
                    f"budget (eps={self.budget.epsilon_max}, delta={self.budget.delta_max}); "
                    f"scope={self.scope_id}"
                )

            conn.execute(
                "INSERT INTO privacy_spend(scope_id, release_id, mechanism, epsilon, delta) "
                "VALUES(?,?,?,?,?)",
                (self.scope_id, release_id, str(mechanism), float(epsilon), float(delta)),
            )
            conn.execute("COMMIT")
            return PrivacySpend(str(mechanism), float(epsilon), float(delta))
        except Exception:
            try:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
            finally:
                conn.close()
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def spend(self, mechanism: str, epsilon: float, delta: float = 0.0) -> PrivacySpend:
        return self._atomic_spend(mechanism, epsilon, delta, release_id=None)

    def spend_once(self, release_id: str, mechanism: str, epsilon: float, delta: float = 0.0) -> PrivacySpend:
        return self._atomic_spend(mechanism, epsilon, delta, release_id=str(release_id))

    def statement(self) -> dict:
        eps, delta, count = self._totals()
        return {
            "accountant": "sqlite_basic_sequential_composition",
            "scope_id": self.scope_id,
            "epsilon_spent": eps,
            "delta_spent": delta,
            "epsilon_max": self.budget.epsilon_max,
            "delta_max": self.budget.delta_max,
            "release_count": count,
            "persistent": True,
            "atomic_check_and_spend": True,
            "guarantee": "Composed releases are (sum epsilon_i, sum delta_i)-DP under the stated per-mechanism adjacency assumptions; ledger persistence prevents restart budget reset for this scope.",
        }
