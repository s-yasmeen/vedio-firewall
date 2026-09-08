from pathlib import Path

import pytest

from tapf.formal_privacy import FormalPrivacyBudget, PrivacyBudgetExceeded
from tapf.persistent_privacy import DuplicateRelease, SQLiteDPAccountant


def test_persists_across_restart(tmp_path: Path):
    db = tmp_path / "privacy.sqlite3"
    budget = FormalPrivacyBudget(1.0, 0.0)
    a = SQLiteDPAccountant(db, "patient-session-1", budget)
    a.spend_once("r1", "rr", 0.4, 0.0)

    b = SQLiteDPAccountant(db, "patient-session-1", budget)
    assert b.epsilon_spent == pytest.approx(0.4)
    assert b.statement()["release_count"] == 1


def test_scopes_are_isolated(tmp_path: Path):
    db = tmp_path / "privacy.sqlite3"
    budget = FormalPrivacyBudget(1.0, 0.0)
    a = SQLiteDPAccountant(db, "scope-a", budget)
    b = SQLiteDPAccountant(db, "scope-b", budget)
    a.spend("rr", 0.7, 0.0)
    assert a.epsilon_spent == pytest.approx(0.7)
    assert b.epsilon_spent == pytest.approx(0.0)


def test_fail_closed_after_restart(tmp_path: Path):
    db = tmp_path / "privacy.sqlite3"
    budget = FormalPrivacyBudget(1.0, 0.0)
    SQLiteDPAccountant(db, "s", budget).spend_once("r1", "rr", 0.75, 0.0)
    restarted = SQLiteDPAccountant(db, "s", budget)
    with pytest.raises(PrivacyBudgetExceeded):
        restarted.spend_once("r2", "rr", 0.30, 0.0)
    assert restarted.epsilon_spent == pytest.approx(0.75)


def test_duplicate_release_id_is_blocked(tmp_path: Path):
    db = tmp_path / "privacy.sqlite3"
    a = SQLiteDPAccountant(db, "s", FormalPrivacyBudget(2.0, 0.0))
    a.spend_once("same-id", "rr", 0.5, 0.0)
    with pytest.raises(DuplicateRelease):
        a.spend_once("same-id", "rr", 0.5, 0.0)
    assert a.epsilon_spent == pytest.approx(0.5)
