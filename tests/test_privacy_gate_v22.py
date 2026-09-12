import pytest

from tapf.privacy_gate_v22 import (
    TemporalExposureAccumulator,
    effective_auc,
    effective_auc_upper_from_ci,
    evaluate_v22_gate,
)


def test_effective_auc_is_chance_centered_and_symmetric():
    assert effective_auc(0.50) == pytest.approx(0.50)
    assert effective_auc(0.40) == pytest.approx(0.60)
    assert effective_auc(0.60) == pytest.approx(0.60)
    assert effective_auc(0.45) == pytest.approx(0.55)
    assert effective_auc(0.55) == pytest.approx(0.55)


def test_ci_gate_uses_farthest_endpoint_from_chance():
    assert effective_auc_upper_from_ci([0.47, 0.54]) == pytest.approx(0.54)
    assert effective_auc_upper_from_ci([0.42, 0.53]) == pytest.approx(0.58)


def test_v22_gate_releases_only_when_every_condition_passes():
    ok = evaluate_v22_gate(
        clip_auc_ci95=[0.47, 0.54],
        repeated_release_auc=0.53,
        task_f1_ci95=[0.21, 0.30],
        attacker_aucs=[0.51, 0.54, 0.48, 0.52],
    )
    assert ok["release_eligible"] is True
    assert ok["decision"] == "RELEASE"

    bad = evaluate_v22_gate(
        clip_auc_ci95=[0.47, 0.54],
        repeated_release_auc=0.58,
        task_f1_ci95=[0.21, 0.30],
        attacker_aucs=[0.51, 0.54, 0.48, 0.52],
    )
    assert bad["release_eligible"] is False
    assert bad["decision"] == "BLOCK"


def test_auc_below_half_is_not_automatically_private():
    result = evaluate_v22_gate(
        clip_auc_ci95=[0.39, 0.43],
        repeated_release_auc=0.40,
        task_f1_ci95=[0.25, 0.35],
        attacker_aucs=[0.40],
    )
    assert result["release_eligible"] is False


def test_temporal_accumulator_fails_closed_after_budget():
    acc = TemporalExposureAccumulator(max_total_advantage=0.10)
    acc.add_auc(0.53)
    assert acc.allow_more is True
    acc.add_auc(0.54)
    assert acc.allow_more is True
    acc.add_auc(0.55)
    assert acc.allow_more is False
    acc.reset()
    assert acc.total_advantage == 0.0
