"""Portable edge benchmark harness for TAPF-MIN release-service path.

Measures wall-clock latency and process RSS for repeated authorization+gate evaluations.
This is an engineering benchmark, not a mobile battery benchmark or clinical validation.
"""
from __future__ import annotations
import argparse, json, os, statistics, time
from pathlib import Path

try:
    import resource
except Exception:
    resource = None

from tapf.deployment_v22 import V22BoundedEvidence, select_authorized_v22_release
from tapf.task_contract import TaskAuthorizationContract


def rss_mb():
    if resource is None:
        return None
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return v / (1024.0 if v < 10**9 else 1024.0**2)


def one_eval():
    now = time.time()
    from datetime import datetime, timezone, timedelta
    t = datetime.now(timezone.utc)
    contract = TaskAuthorizationContract(
        session_id="edge-bench-session",
        task="movement",
        purpose="movement assessment benchmark",
        recipient_id="edge-bench-clinic",
        requested_representation="motion-features",
        patient_authorized=True,
        issued_at=(t - timedelta(seconds=1)).isoformat(),
        expires_at=(t + timedelta(minutes=5)).isoformat(),
    )
    evidence = [V22BoundedEvidence(
        alpha=0.2,
        clip_auc_ci95_low=0.49,
        clip_auc_ci95_high=0.54,
        repeated_release_auc=0.53,
        task_f1_ci95_low=0.24,
        task_f1_ci95_high=0.31,
        attacker_aucs=(0.51,0.52,0.53),
        latency_ms=40.0,
        evaluator_id="edge-benchmark",
        sample_count=100,
    )]
    return select_authorized_v22_release(contract, evidence, expected_recipient_id="edge-bench-clinic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--out", default="results/edge_benchmark.json")
    args = ap.parse_args()
    times=[]
    for _ in range(args.iterations):
        t0=time.perf_counter(); result=one_eval(); times.append((time.perf_counter()-t0)*1000)
        if result["decision"] != "RELEASE":
            raise RuntimeError("benchmark control evaluation did not RELEASE")
    times_sorted=sorted(times)
    def pct(p):
        i=min(len(times_sorted)-1, max(0, int(round((p/100)*(len(times_sorted)-1)))))
        return times_sorted[i]
    out={
        "scope":"Authorization + empirical release-controller path only; excludes video decoding/model inference and battery measurement.",
        "iterations":args.iterations,
        "latency_ms":{"mean":statistics.mean(times),"median":statistics.median(times),"p95":pct(95),"p99":pct(99),"max":max(times)},
        "process_peak_rss_mb":rss_mb(),
        "python_pid":os.getpid(),
    }
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__ == "__main__":
    main()
