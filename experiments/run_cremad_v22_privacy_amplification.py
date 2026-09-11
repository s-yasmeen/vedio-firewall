"""TAPF-MIN v2.2: privacy amplification and repeated-release benchmark.

This experiment is prospective relative to the v2.1 benchmark. It does not rewrite
historical thresholds or results. It evaluates a stricter, chance-centered privacy gate
and explicitly targets repeated-release leakage.

Scientific scope:
- CREMA-D DFA emotion utility benchmark,
- actor-disjoint cross-fitting,
- 147-D motion descriptor input,
- adversarial task representation inherited from v2.1,
- deterministic minimum-disclosure transforms,
- stochastic randomized-response release candidates,
- clip and repeated-release identity attacks,
- worst-case attacker logic,
- bootstrap confidence intervals,
- fail-closed v2.2 release decision.

This remains a controlled research benchmark, not clinical validation and not a formal
anonymity or differential-privacy guarantee.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import numpy as np
from sklearn.metrics import f1_score

from tapf.minimum_representation import read_motion_clip
from tapf.privacy_gate_v22 import effective_auc, evaluate_v22_gate
from experiments.run_cremad_v21_adversarial import (
    EMOTIONS,
    crossfit,
    quantize_probabilities,
)
from experiments.run_cremad_final_validation import (
    _attack,
    repeated_release_attack,
)
from experiments.run_cremad_privacy_utility_spectrum import bootstrap_metric

NAME_RE = re.compile(r"^(\d{4})_DFA_(ANG|DIS|FEA|HAP|NEU|SAD)_XX\.flv$", re.I)
MIN_REAL_VIDEO_BYTES = 1000


def temperature_flatten(P: np.ndarray, temperature: float) -> np.ndarray:
    """Flatten posterior confidence while preserving its ordering."""
    P = np.asarray(P, np.float64)
    eps = 1e-8
    logits = np.log(np.clip(P, eps, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    out /= out.sum(axis=1, keepdims=True)
    return out.astype(np.float32)


def hard_label_release(P: np.ndarray) -> np.ndarray:
    idx = np.argmax(P, axis=1)
    return np.eye(P.shape[1], dtype=np.float32)[idx]


def randomized_response_release(P: np.ndarray, keep_probability: float, seed: int) -> np.ndarray:
    """Randomize the released task label to reduce persistent per-person error signatures.

    This is an empirical privacy-amplification candidate. We do not label it as formal
    local differential privacy in this benchmark because the complete end-to-end mechanism
    and composition accounting are not being claimed here.
    """
    if not 0.0 <= keep_probability <= 1.0:
        raise ValueError("keep_probability must be in [0,1]")
    rng = np.random.default_rng(seed)
    n, k = P.shape
    base = np.argmax(P, axis=1)
    out = base.copy()
    flip = rng.random(n) > keep_probability
    for i in np.flatnonzero(flip):
        alternatives = np.arange(k)
        alternatives = alternatives[alternatives != base[i]]
        out[i] = int(rng.choice(alternatives))
    return np.eye(k, dtype=np.float32)[out]


def _task_metrics(Z: np.ndarray, emotion: np.ndarray, seed: int) -> tuple[float, float, float]:
    pred = EMOTIONS[np.argmax(Z, axis=1)]
    return bootstrap_metric(
        emotion,
        pred,
        lambda y, p: f1_score(y, p, average="macro", zero_division=0),
        seed=seed,
        n=400,
    )


def evaluate_once(Z, actor, emotion, seed: int, label: str) -> dict:
    f1, lo, hi = _task_metrics(Z, emotion, seed)
    clip = _attack(Z, actor, seed + 500)
    repeated = repeated_release_attack(Z, actor, seed + 900)
    attacker_aucs = [float(r["auc"]) for r in clip["attackers"]]
    gate = evaluate_v22_gate(
        clip_auc_ci95=clip["identity_auc_ci95"],
        repeated_release_auc=repeated["identity_auc"],
        task_f1_ci95=[lo, hi],
        attacker_aucs=attacker_aucs,
    )
    return {
        "release": label,
        "dimension": int(Z.shape[1]),
        "emotion_macro_f1": float(f1),
        "emotion_macro_f1_ci95": [float(lo), float(hi)],
        "clip_identity": clip,
        "repeated_release_identity": repeated,
        "clip_effective_auc": float(effective_auc(clip["identity_auc"])),
        "repeated_effective_auc": float(effective_auc(repeated["identity_auc"])),
        "v22_gate": gate,
    }


def evaluate_stochastic(factory, actor, emotion, seed: int, label: str, replicates: int = 5) -> dict:
    """Evaluate stochastic release conservatively across predeclared replicate seeds."""
    rows = []
    for r in range(replicates):
        Z = factory(seed + 10007 * r)
        rows.append(evaluate_once(Z, actor, emotion, seed + 20011 * r, label))

    # Fail-closed aggregation: privacy uses worst replicate; utility uses worst lower CI.
    worst_clip = max(rows, key=lambda x: x["v22_gate"]["observed"]["clip_effective_auc_upper"])
    worst_repeat = max(rows, key=lambda x: x["repeated_effective_auc"])
    worst_utility = min(rows, key=lambda x: x["emotion_macro_f1_ci95"][0])
    release_eligible = all(r["v22_gate"]["release_eligible"] for r in rows)
    return {
        "release": label,
        "stochastic_replicates": replicates,
        "release_eligible_all_replicates": release_eligible,
        "replicates": rows,
        "conservative_summary": {
            "max_clip_effective_auc_upper": float(worst_clip["v22_gate"]["observed"]["clip_effective_auc_upper"]),
            "max_repeated_effective_auc": float(worst_repeat["repeated_effective_auc"]),
            "min_task_f1_lower_ci": float(worst_utility["emotion_macro_f1_ci95"][0]),
            "decision": "RELEASE" if release_eligible else "BLOCK",
        },
    }


def run(root, folds: int = 5, seed: int = 42):
    root = Path(root)
    files = []
    for p in sorted(root.glob("*.flv")):
        m = NAME_RE.match(p.name)
        if m and p.stat().st_size > MIN_REAL_VIDEO_BYTES:
            files.append((p, m.group(1), m.group(2).upper()))

    actor = np.asarray([a for _, a, _ in files])
    emotion = np.asarray([e for _, _, e in files])
    if len(files) < 120 or len(np.unique(actor)) < 8:
        raise RuntimeError(f"Insufficient hydrated CREMA-D cohort: {len(files)} clips/{len(np.unique(actor))} actors")

    X = np.asarray([read_motion_clip(str(p)) for p, _, _ in files], np.float32)
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite motion features")
    folds = max(2, min(int(folds), len(np.unique(actor))))

    # Predeclared sweep. We keep it compact enough to reproduce while covering the
    # strongest v2.1 region and explicit repeated-release amplification candidates.
    adversarial_strengths = [0.0, 0.10, 0.15, 0.25]
    rows = []
    candidates = []

    for i, strength in enumerate(adversarial_strengths):
        P, fold_rows = crossfit(X, actor, emotion, folds, strength, seed + 1000 * i)
        deterministic = [
            ("posterior-6d", P),
            ("posterior-q8", quantize_probabilities(P, 8)),
            ("posterior-q4", quantize_probabilities(P, 4)),
            ("posterior-q2", quantize_probabilities(P, 2)),
            ("task-label-onehot", hard_label_release(P)),
            ("posterior-temp-1.5", temperature_flatten(P, 1.5)),
            ("posterior-temp-2.0", temperature_flatten(P, 2.0)),
            ("posterior-temp-4.0", temperature_flatten(P, 4.0)),
        ]
        det_rows = []
        for j, (name, Z) in enumerate(deterministic):
            ev = evaluate_once(Z, actor, emotion, seed + 5000 * i + 100 * j, name)
            ev["adversarial_strength"] = strength
            det_rows.append(ev)
            candidates.append({
                "kind": "deterministic",
                "adversarial_strength": strength,
                "release": name,
                "release_eligible": ev["v22_gate"]["release_eligible"],
                "clip_effective_auc_upper": ev["v22_gate"]["observed"]["clip_effective_auc_upper"],
                "repeated_effective_auc": ev["repeated_effective_auc"],
                "task_f1_lower_ci": ev["emotion_macro_f1_ci95"][0],
                "task_f1": ev["emotion_macro_f1"],
            })

        stochastic_rows = []
        for j, keep in enumerate([0.90, 0.80, 0.70, 0.60, 0.50]):
            label = f"task-label-randomized-response-p{keep:.2f}"
            ev = evaluate_stochastic(
                lambda local_seed, P=P, keep=keep: randomized_response_release(P, keep, local_seed),
                actor,
                emotion,
                seed + 9000 * i + 200 * j,
                label,
                replicates=5,
            )
            ev["adversarial_strength"] = strength
            ev["keep_probability"] = keep
            stochastic_rows.append(ev)
            s = ev["conservative_summary"]
            candidates.append({
                "kind": "stochastic",
                "adversarial_strength": strength,
                "release": label,
                "release_eligible": ev["release_eligible_all_replicates"],
                "clip_effective_auc_upper": s["max_clip_effective_auc_upper"],
                "repeated_effective_auc": s["max_repeated_effective_auc"],
                "task_f1_lower_ci": s["min_task_f1_lower_ci"],
                "task_f1": float(np.mean([r["emotion_macro_f1"] for r in ev["replicates"]])),
            })

        rows.append({
            "adversarial_strength": strength,
            "folds": fold_rows,
            "deterministic_releases": det_rows,
            "stochastic_releases": stochastic_rows,
        })

    eligible = [c for c in candidates if c["release_eligible"]]
    # Predeclared ranking: minimize the worst privacy excess above chance first,
    # then maximize conservative utility. No post-hoc threshold relaxation.
    def rank_key(c):
        privacy = max(c["clip_effective_auc_upper"], c["repeated_effective_auc"])
        return (privacy, -c["task_f1_lower_ci"], -c["task_f1"])

    best_overall = min(candidates, key=rank_key)
    best_eligible = min(eligible, key=rank_key) if eligible else None
    best_utility = max(candidates, key=lambda c: (c["task_f1_lower_ci"], c["task_f1"]))

    result = {
        "dataset": "CREMA-D DFA cohort",
        "clips": len(files),
        "actors": int(len(np.unique(actor))),
        "model": "TAPF-MIN v2.2 privacy-amplified feature-level benchmark",
        "scientific_scope": "Controlled emotion-utility benchmark on 147-D motion input; not raw-video end-to-end clinical validation and not formal anonymity.",
        "protocol": {
            "utility": f"{folds}-fold actor-disjoint out-of-fold emotion prediction",
            "privacy": "four independent post-hoc identity attackers; chance-centered effective AUC",
            "repeated_release": "identity attack on disjoint per-actor aggregated releases",
            "stochastic_robustness": "five predeclared replicate seeds; all replicates must pass",
            "release_gate": "clip effective-AUC upper bound <= 0.55 AND repeated-release effective AUC <= 0.55 AND every independent attacker <= 0.55 AND lower emotion Macro-F1 CI >= 0.20",
        },
        "targets": {
            "chance_identity_auc": 0.50,
            "max_identity_advantage": 0.05,
            "max_effective_auc": 0.55,
            "minimum_lower_ci_emotion_f1": 0.20,
        },
        "sweep": rows,
        "candidate_table": candidates,
        "eligible_operating_points": eligible,
        "best_overall_by_predeclared_ranking": best_overall,
        "best_eligible_by_predeclared_ranking": best_eligible,
        "best_conservative_utility": best_utility,
        "integrity": {
            "historical_v21_thresholds_rewritten": False,
            "threshold_relaxation_after_results": False,
            "actor_disjoint_utility": all(fr["actor_overlap"] == 0 for r in rows for fr in r["folds"]),
            "repeated_release_attack": True,
            "chance_centered_auc": True,
            "fail_closed": True,
        },
    }
    Path("results").mkdir(exist_ok=True)
    out = Path("results/cremad_v22_privacy_amplification.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.root, args.folds, args.seed)
