"""Executable TAPF-MIN controller demonstration.

Outputs are SIMULATION ONLY and must not be reported as empirical research results.
"""
import json
from pathlib import Path
from tapf import ReleasePolicy, MinimumDisclosureController, build_attestation
from tapf.simulation import evaluate_simulated


def main():
    output = Path("results")
    output.mkdir(exist_ok=True)

    policy = ReleasePolicy(
        privacy_threshold=0.25,
        utility_threshold=0.75,
        temporal_threshold=0.30,
    )
    controller = MinimumDisclosureController(policy)
    decision = controller.search(evaluate_simulated)
    attestation = build_attestation(
        decision,
        policy,
        task="facial-expression-utility-demo",
        representation="adaptive-protected-video",
        evaluator_versions={"identity": "simulation-v1", "utility": "simulation-v1"},
    )

    (output / "simulation_decision.json").write_text(json.dumps(decision, indent=2))
    (output / "simulation_attestation.json").write_text(json.dumps(attestation, indent=2))

    print(json.dumps(attestation, indent=2))
    print("NOTE: simulation output; not empirical evidence.")


if __name__ == "__main__":
    main()
