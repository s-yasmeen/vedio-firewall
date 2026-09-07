"""Build a TAPF-MIN privacy-utility operating frontier from measured metrics.

Input CSV/JSON must contain one record per alpha with:
alpha, identity_risk, task_utility, temporal_risk.
The script never fabricates missing measurements.
"""
from pathlib import Path
import argparse, json
import pandas as pd
from tapf import ReleasePolicy, MinimumDisclosureController, build_attestation


def load_records(path: str):
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p).to_dict("records")
    return json.loads(p.read_text())


def run(path: str, privacy=0.25, utility=0.75, temporal=0.30):
    rows = sorted(load_records(path), key=lambda r: float(r["alpha"]))
    by_alpha = {float(r["alpha"]): r for r in rows}
    alphas = tuple(sorted(by_alpha))
    policy = ReleasePolicy(
        privacy_threshold=privacy,
        utility_threshold=utility,
        temporal_threshold=temporal,
        alphas=alphas,
        fail_closed=True,
    )

    def evaluator(alpha):
        r = by_alpha[float(alpha)]
        return float(r["identity_risk"]), float(r["task_utility"]), float(r["temporal_risk"])

    controller = MinimumDisclosureController(policy)
    decision = controller.search(evaluator)
    attestation = build_attestation(
        decision,
        policy,
        task="measured-downstream-task",
        representation="adaptive-protected-video",
        evaluator_versions={"source": str(path), "mode": "measured-input"},
    )
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "measured_frontier_decision.json").write_text(json.dumps(decision, indent=2))
    (out / "measured_frontier_attestation.json").write_text(json.dumps(attestation, indent=2))
    print(json.dumps(attestation, indent=2))
    return decision


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("--privacy", type=float, default=0.25)
    ap.add_argument("--utility", type=float, default=0.75)
    ap.add_argument("--temporal", type=float, default=0.30)
    args = ap.parse_args()
    run(args.metrics, args.privacy, args.utility, args.temporal)
