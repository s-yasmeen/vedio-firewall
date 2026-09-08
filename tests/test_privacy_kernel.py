import numpy as np
import pytest

from tapf.formal_privacy import PrivacyBudgetExceeded
from tapf.privacy_kernel import MinimumDisclosurePrivacyKernel, ReleaseEvidence


CLASSES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]


def test_kernel_prefers_three_bit_private_label_when_it_meets_frozen_evidence():
    kernel = MinimumDisclosurePrivacyKernel(CLASSES, epsilon_budget=3.0, delta_budget=1e-5)
    evidence = [
        ReleaseEvidence("label-e1", "randomized_response_label", 0.76, 0.07, 1.0),
        ReleaseEvidence("posterior-e1", "gaussian_quantized_posterior", 0.88, 0.06, 1.0, 1e-5, 8),
    ]
    out = kernel.release(
        [0.05, 0.05, 0.05, 0.75, 0.05, 0.05], evidence,
        min_utility_lower=0.75, max_empirical_identity_risk_upper=0.10,
        rng=np.random.default_rng(9),
    )
    assert out["decision"] == "RELEASE"
    assert out["selected"] == "label-e1"
    assert out["disclosure_bits"] == 3
    assert out["payload"]["type"] == "categorical_label"
    assert out["raw_video_released"] is False
    assert out["latent_released"] is False
    assert out["formal_privacy"]["epsilon_spent"] == pytest.approx(1.0)


def test_kernel_uses_dp_posterior_only_when_label_utility_is_insufficient():
    kernel = MinimumDisclosurePrivacyKernel(CLASSES, epsilon_budget=2.0, delta_budget=1e-4)
    evidence = [
        ReleaseEvidence("label", "randomized_response_label", 0.55, 0.05, 0.8),
        ReleaseEvidence("posterior", "gaussian_quantized_posterior", 0.80, 0.08, 1.0, 1e-5, 4),
    ]
    out = kernel.release(
        [0.05, 0.10, 0.05, 0.70, 0.05, 0.05], evidence,
        min_utility_lower=0.75, max_empirical_identity_risk_upper=0.10,
        rng=np.random.default_rng(11),
    )
    assert out["decision"] == "RELEASE"
    assert out["selected"] == "posterior"
    assert out["disclosure_bits"] == 24
    assert out["payload"]["type"] == "quantized_posterior"
    assert sum(out["payload"]["values"]) == pytest.approx(1.0, abs=1e-5)
    assert out["mechanism"]["post_processing"] == "simplex_projection_then_quantization"


def test_kernel_blocks_when_frozen_evidence_does_not_meet_joint_gate():
    kernel = MinimumDisclosurePrivacyKernel(CLASSES, epsilon_budget=3.0)
    evidence = [ReleaseEvidence("label", "randomized_response_label", 0.60, 0.20, 1.0)]
    out = kernel.release(
        [1, 0, 0, 0, 0, 0], evidence,
        min_utility_lower=0.75, max_empirical_identity_risk_upper=0.10,
    )
    assert out["decision"] == "BLOCK"
    assert kernel.accountant.epsilon_spent == 0.0


def test_repeated_releases_consume_composed_budget_and_then_fail_closed():
    kernel = MinimumDisclosurePrivacyKernel(CLASSES, epsilon_budget=1.5)
    evidence = [ReleaseEvidence("label", "randomized_response_label", 0.90, 0.05, 0.75)]
    posterior = [0.7, 0.1, 0.05, 0.05, 0.05, 0.05]
    first = kernel.release(
        posterior, evidence, min_utility_lower=0.80,
        max_empirical_identity_risk_upper=0.10, rng=np.random.default_rng(1),
    )
    second = kernel.release(
        posterior, evidence, min_utility_lower=0.80,
        max_empirical_identity_risk_upper=0.10, rng=np.random.default_rng(2),
    )
    third = kernel.release(
        posterior, evidence, min_utility_lower=0.80,
        max_empirical_identity_risk_upper=0.10, rng=np.random.default_rng(3),
    )
    assert first["decision"] == "RELEASE"
    assert second["decision"] == "RELEASE"
    assert second["formal_privacy"]["epsilon_spent"] == pytest.approx(1.5)
    assert third["decision"] == "BLOCK"
    assert third["selected"] is None


def test_gaussian_evidence_cannot_claim_zero_delta():
    kernel = MinimumDisclosurePrivacyKernel(CLASSES, epsilon_budget=2.0, delta_budget=1e-3)
    evidence = [ReleaseEvidence("posterior", "gaussian_quantized_posterior", 0.9, 0.05, 1.0, 0.0, 8)]
    with pytest.raises(ValueError, match="Gaussian mechanism requires delta > 0"):
        kernel.release(
            [0.7, .1, .05, .05, .05, .05], evidence,
            min_utility_lower=.8, max_empirical_identity_risk_upper=.1,
        )
