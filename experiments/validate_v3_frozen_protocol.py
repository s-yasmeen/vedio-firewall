"""Fail-closed validator for the frozen TAPF-MIN v3 final protocol.

This script does not train a model or tune thresholds. It consumes final-result JSON and
the pre-frozen protocol, evaluates every declared gate, and emits RELEASE_ELIGIBLE only
if every required condition is present and passes. Missing evidence is a BLOCK.
"""
from __future__ import annotations

from pathlib import Path
import argparse, hashlib, json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def validate(protocol_path: str | Path, result_path: str | Path):
    protocol_path = Path(protocol_path)
    result_path = Path(result_path)
    protocol = json.loads(protocol_path.read_text())
    result = json.loads(result_path.read_text())
    gates = []

    def gate(name, observed, rule, passed):
        gates.append({"name": name, "observed": observed, "rule": rule, "pass": bool(passed)})

    utility = protocol["utility_gate"]
    ci = _get(result, "utility", "actor_cluster_macro_f1_ci95")
    acc = _get(result, "utility", "accuracy")
    gate("utility_macro_f1_lower_ci", None if ci is None else ci[0],
         f">={utility['minimum']}", ci is not None and ci[0] >= utility["minimum"])
    gate("utility_accuracy", acc, f">={utility['secondary_accuracy_minimum']}",
         acc is not None and acc >= utility["secondary_accuracy_minimum"])

    p = _get(result, "protocol") or {}
    gate("actor_overlap", p.get("actor_overlap"), "==0", p.get("actor_overlap") == 0)
    gate("single_frozen_model", p.get("single_frozen_fer_checkpoint_for_audit"), "==true",
         p.get("single_frozen_fer_checkpoint_for_audit") is True)
    gate("audit_actor_count", p.get("privacy_audit_actors"),
         f">={protocol['dataset']['minimum_privacy_audit_actors']}",
         isinstance(p.get("privacy_audit_actors"), int) and
         p.get("privacy_audit_actors") >= protocol["dataset"]["minimum_privacy_audit_actors"])

    ep = _get(result, "empirical_privacy") or {}
    g = protocol["empirical_privacy_gates"]
    closed = _get(ep, "closed_set_identity", "effective_auc_upper_ci95")
    if closed is None:
        closed = _get(ep, "closed_set_identity", "effective_auc")
    pairwise = _get(ep, "pairwise_linkage", "effective_auc")
    centroid = _get(ep, "centroid_linkage", "accuracy_advantage")
    repeated = _get(ep, "repeated_release_identity", "effective_auc")
    gate("closed_set_identity", closed, f"<={g['closed_set_effective_auc_upper']}",
         closed is not None and closed <= g["closed_set_effective_auc_upper"])
    gate("pairwise_linkage", pairwise, f"<={g['pairwise_linkage_effective_auc_upper']}",
         pairwise is not None and pairwise <= g["pairwise_linkage_effective_auc_upper"])
    gate("centroid_linkage", centroid, f"<={g['centroid_linkage_accuracy_advantage_upper']}",
         centroid is not None and centroid <= g["centroid_linkage_accuracy_advantage_upper"])
    gate("repeated_release", repeated, f"<={g['repeated_release_effective_auc_upper']}",
         repeated is not None and repeated <= g["repeated_release_effective_auc_upper"])

    formal = _get(result, "formal_private_release")
    fp = protocol["formal_privacy"]
    gate("formal_release_evidence_present", formal is not None, "==true", formal is not None)
    if formal:
        mechanism = formal.get("mechanism")
        epsilon = formal.get("epsilon_per_release")
        delta = formal.get("delta_per_release")
        private_utility = formal.get("private_release_utility")
        gate("formal_mechanism", mechanism, f"in {fp['allowed_mechanisms']}",
             mechanism in fp["allowed_mechanisms"])
        gate("epsilon_per_release", epsilon, f"<={fp['epsilon_per_release_max']}",
             epsilon is not None and 0 < epsilon <= fp["epsilon_per_release_max"])
        gate("delta_per_release", delta, f"<={fp['delta_per_release_max']}",
             delta is not None and 0 <= delta <= fp["delta_per_release_max"])
        gate("private_release_utility_measured", private_utility, "present", private_utility is not None)
        gate("persistent_accounting", formal.get("persistent_accounting"), "==true",
             formal.get("persistent_accounting") is True)
        gate("duplicate_release_protection", formal.get("duplicate_release_protection"), "==true",
             formal.get("duplicate_release_protection") is True)

    disclosure = _get(result, "minimum_disclosure")
    gate("minimum_disclosure_evidence_present", disclosure is not None, "==true", disclosure is not None)
    if disclosure:
        gate("raw_video_not_released", disclosure.get("raw_video_released"), "==false",
             disclosure.get("raw_video_released") is False)
        gate("task_latent_not_released", disclosure.get("task_latent_released"), "==false",
             disclosure.get("task_latent_released") is False)
        bits = disclosure.get("disclosure_bits")
        gate("disclosure_bits_declared", bits, ">0", isinstance(bits, int) and bits > 0)

    passed = all(x["pass"] for x in gates)
    report = {
        "schema": "tapf-min-v3-frozen-validation/v1",
        "decision": "RELEASE_ELIGIBLE" if passed else "BLOCK",
        "protocol_sha256": _sha256(protocol_path),
        "result_sha256": _sha256(result_path),
        "gate_count": len(gates),
        "failed_gate_count": sum(not x["pass"] for x in gates),
        "gates": gates,
        "scope": protocol.get("scientific_scope"),
        "claims": {"clinical_validation": False, "anonymity": False},
    }
    Path("results").mkdir(exist_ok=True)
    out = Path("results/tapf_min_v3_frozen_validation.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--protocol", default="protocols/tapf_min_v3_final_protocol.json")
    a = ap.parse_args()
    validate(a.protocol, a.result)
