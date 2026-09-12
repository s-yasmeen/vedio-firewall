"""Export a compact, reproducible TAPF-MIN final-validation table from benchmark JSON."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def rows_from_result(result):
    rows=[]
    for sweep in result.get("sweep",[]):
        lam=sweep["adversarial_strength"]
        for e in sweep.get("releases",[]):
            clip=e["clip_identity"]; seq=e["repeated_release_identity"]
            eligible=bool(e.get("privacy_gate_upper_ci_le_0_60") and
                          e.get("temporal_gate_auc_le_0_60") and
                          e.get("utility_gate_lower_ci_ge_0_20"))
            rows.append({
                "lambda":lam,
                "release":e["release"],
                "dimension":e["dimension"],
                "emotion_macro_f1":e["emotion_macro_f1"],
                "emotion_f1_ci_low":e["emotion_macro_f1_ci95"][0],
                "emotion_f1_ci_high":e["emotion_macro_f1_ci95"][1],
                "clip_identity_auc":clip["identity_auc"],
                "clip_identity_auc_ci_low":clip["identity_auc_ci95"][0],
                "clip_identity_auc_ci_high":clip["identity_auc_ci95"][1],
                "clip_worst_attacker":clip["worst_case_attacker"],
                "repeated_release_identity_auc":seq["identity_auc"],
                "repeated_worst_attacker":seq["worst_case_attacker"],
                "conservative_release_eligible":eligible,
            })
    rows.sort(key=lambda r:(not r["conservative_release_eligible"],
                            max(r["clip_identity_auc"],r["repeated_release_identity_auc"]),
                            -r["emotion_macro_f1"]))
    return rows


def run(src, out=None):
    src=Path(src); result=json.loads(src.read_text()); rows=rows_from_result(result)
    out=Path(out) if out else src.with_suffix(".summary.csv")
    if not rows: raise RuntimeError("No validation rows found")
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} operating points to {out}")
    print("Top ranked operating point:")
    print(json.dumps(rows[0],indent=2))
    return rows

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("json"); ap.add_argument("--out")
    a=ap.parse_args(); run(a.json,a.out)
