from tapf import ReleasePolicy, MinimumDisclosureController
from tapf.simulation import evaluate_simulated
from tapf.temporal_privacy import accumulated_identity_evidence


def test_temporal_risk_nonnegative():
    assert accumulated_identity_evidence([0.1, 0.2, 0.1]) >= 0


def test_controller_returns_valid_decision():
    policy = ReleasePolicy()
    result = MinimumDisclosureController(policy).search(evaluate_simulated)
    assert result["decision"] in {"RELEASE", "BLOCK", "UNVERIFIED"}
    assert len(result["history"]) > 0


def test_fail_closed_when_impossible():
    policy = ReleasePolicy(privacy_threshold=0.0, utility_threshold=1.0, temporal_threshold=0.0)
    result = MinimumDisclosureController(policy).search(evaluate_simulated)
    assert result["decision"] == "BLOCK"
