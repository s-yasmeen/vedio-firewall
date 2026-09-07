"""Machine-readable empirical release attestation."""
from datetime import datetime, timezone


def build_attestation(decision: dict, policy, task: str,
                      representation: str, evaluator_versions=None):
    selected = decision.get("evaluation") or {}
    return {
        "schema": "tapf-min-attestation/v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "representation": representation,
        "release_decision": decision["decision"],
        "selected_alpha": decision.get("selected_alpha"),
        "identity_risk": selected.get("identity_risk"),
        "task_utility": selected.get("task_utility"),
        "temporal_risk": selected.get("temporal_risk"),
        "thresholds": {
            "privacy": policy.privacy_threshold,
            "utility": policy.utility_threshold,
            "temporal": policy.temporal_threshold,
        },
        "raw_biometric_transmitted": False,
        "evaluator_versions": evaluator_versions or {},
        "scope": "Empirical result under the tested models and threat model; not a formal anonymity or security certificate.",
    }
