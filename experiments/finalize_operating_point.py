"""Finalize a TAPF-MIN operating point from measured experiment records.

Expected JSON input: a list of rows containing alpha, identity_risk, task_utility,
and temporal_risk. No score is simulated or inferred by this script.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from tapf import ReleasePolicy, MinimumDisclosureController, build_attestation


def load_rows(path):
    rows = json.loads(Path(path).read_text())
    required = {"alpha", "identity_risk", "task_utility", "temporal_risk"}
    for i, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Row {i} missing: {sorted(missing)}")
    return sorted(rows, key=lambda r: float(r["alpha"]))


def finalize(metrics_path, output="results/final_attestation.json",
             privacy=0.25, utility=0.75, temporal=0.30,
             task="facial-expression", representation="protected-video"):
    rows = load_rows(metrics_path)
    lookup = {round(float(r["alpha"]), 8): r for r in rows}
    alphas = tuple(sorted(lookup))
    policy = ReleasePolicy(
        privacy_threshold=privacy,
        utility_threshold=utility,
        temporal_threshold=temporal,
        alphas=alphas,
        fail_closed=True,
    )

    def evaluator(alpha):
        r = lookup[round(float(alpha), 8)]
        return float(r["identity_risk"]), float(r["task_utility"]), float(r["temporal_risk"])

    decision = MinimumDisclosureController(policy).search(evaluator)
    attestation = build_attestation(
        decision, policy, task=task, representation=representation,
        evaluator_versions={"source": str(metrics_path), "mode": "measured"},
    )
    payload = {"decision": decision, "attestation": attestation, "source_rows": rows}
    p = Path(output); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("metrics")
    p.add_argument("--output", default="results/final_attestation.json")
    p.add_argument("--privacy", type=float, default=0.25)
    p.add_argument("--utility", type=float, default=0.75)
    p.add_argument("--temporal", type=float, default=0.30)
    p.add_argument("--task", default="facial-expression")
    p.add_argument("--representation", default="protected-video")
    a = p.parse_args()
    finalize(a.metrics, a.output, a.privacy, a.utility, a.temporal, a.task, a.representation)
