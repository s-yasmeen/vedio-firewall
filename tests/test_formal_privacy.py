import math

import numpy as np
import pytest

from tapf.formal_privacy import (
    BasicDPAccountant,
    DisclosureCandidate,
    FormalPrivacyBudget,
    PrivacyBudgetExceeded,
    gaussian_release,
    gaussian_sigma,
    l2_clip,
    randomized_response,
    select_minimum_disclosure,
)


def test_l2_clipping_enforces_public_bound():
    x = l2_clip([3.0, 4.0], clip_norm=1.0)
    assert np.linalg.norm(x) == pytest.approx(1.0)


def test_gaussian_calibration_is_positive_and_scales_with_sensitivity():
    a = gaussian_sigma(epsilon=1.0, delta=1e-5, sensitivity=1.0)
    b = gaussian_sigma(epsilon=1.0, delta=1e-5, sensitivity=2.0)
    assert a > 0
    assert b == pytest.approx(2.0 * a)


def test_gaussian_release_records_formal_parameters_and_spend():
    accountant = BasicDPAccountant(FormalPrivacyBudget(2.0, 2e-5))
    result = gaussian_release(
        [0.8, 0.2], epsilon=1.0, delta=1e-5, clip_norm=1.0,
        rng=np.random.default_rng(7), accountant=accountant,
    )
    assert result["mechanism"] == "gaussian"
    assert result["l2_sensitivity"] == 2.0
    assert result["values"].shape == (2,)
    assert accountant.epsilon_spent == pytest.approx(1.0)
    assert accountant.delta_spent == pytest.approx(1e-5)


def test_gaussian_classic_calibration_rejects_epsilon_above_one():
    with pytest.raises(ValueError, match="epsilon <= 1"):
        gaussian_sigma(epsilon=1.1, delta=1e-5, sensitivity=2.0)


def test_randomized_response_probabilities_satisfy_epsilon_ratio():
    epsilon = 0.8
    result = randomized_response(
        "HAP", ["ANG", "HAP", "NEU"], epsilon=epsilon,
        rng=np.random.default_rng(4),
    )
    ratio = result["truth_probability"] / result["other_probability"]
    assert ratio == pytest.approx(math.exp(epsilon))
    assert result["delta"] == 0.0


def test_basic_composition_fails_closed_when_budget_is_exhausted():
    accountant = BasicDPAccountant(FormalPrivacyBudget(1.0, 1e-5))
    accountant.spend("rr", 0.6, 0.0)
    assert not accountant.can_spend(0.5, 0.0)
    with pytest.raises(PrivacyBudgetExceeded):
        accountant.spend("rr", 0.5, 0.0)


def test_basic_composition_statement_matches_sum_theorem():
    accountant = BasicDPAccountant(FormalPrivacyBudget(2.0, 1e-4))
    accountant.spend("a", 0.4, 1e-5)
    accountant.spend("b", 0.6, 2e-5)
    statement = accountant.statement()
    assert statement["epsilon_spent"] == pytest.approx(1.0)
    assert statement["delta_spent"] == pytest.approx(3e-5)
    assert statement["release_count"] == 2


def test_minimum_disclosure_selects_smallest_eligible_release():
    accountant = BasicDPAccountant(FormalPrivacyBudget(4.0, 1e-5))
    candidates = [
        DisclosureCandidate("dp-label", 3, 0.72, 0.08, 1.0, 0.0),
        DisclosureCandidate("dp-q4-posterior", 12, 0.82, 0.07, 1.0, 0.0),
        DisclosureCandidate("dp-q8-posterior", 24, 0.88, 0.06, 1.0, 0.0),
    ]
    out = select_minimum_disclosure(
        candidates,
        min_utility_lower=0.80,
        max_empirical_identity_risk_upper=0.10,
        accountant=accountant,
    )
    assert out["decision"] == "RELEASE"
    assert out["selected"] == "dp-q4-posterior"
    assert out["disclosure_bits"] == 12


def test_minimum_disclosure_blocks_when_no_candidate_satisfies_joint_constraints():
    accountant = BasicDPAccountant(FormalPrivacyBudget(0.5, 0.0))
    candidates = [DisclosureCandidate("label", 3, 0.95, 0.01, 1.0, 0.0)]
    out = select_minimum_disclosure(
        candidates,
        min_utility_lower=0.90,
        max_empirical_identity_risk_upper=0.10,
        accountant=accountant,
    )
    assert out["decision"] == "BLOCK"
    assert out["selected"] is None
