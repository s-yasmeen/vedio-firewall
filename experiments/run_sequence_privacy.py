"""Sequence-level privacy benchmark for protected video.

Example:
  python experiments/run_sequence_privacy.py video.mp4 --attacker insightface --method static_tapf

The script measures how identity similarity changes as more protected frames are aggregated.
It does not establish anonymity or irreversibility.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np

from tapf.transforms import apply_transform
from attackers.sequence_attack import cumulative_sequence_curve, aggregate_embeddings


def load_attacker(name):
    if name == "insightface":
        from attackers.insightface_attacker import InsightFaceAttacker
        return InsightFaceAttacker()
    if name == "facenet":
        from attackers.facenet_attacker import FaceNetAttacker
        return FaceNetAttacker()
    raise ValueError(name)


def sample_frames(video_path, stride=5, max_frames=60):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames


def run(video_path, attacker_name="insightface", method="static_tapf", alphas=None,
        stride=5, max_frames=60, output="results/sequence_privacy.json"):
    alphas = alphas or [round(i * 0.1, 2) for i in range(11)]
    frames = sample_frames(video_path, stride=stride, max_frames=max_frames)
    if not frames:
        raise RuntimeError("No frames read")
    attacker = load_attacker(attacker_name)

    # Reference identity comes from unprotected frames. Pooling reduces single-frame instability.
    ref_embeddings = [attacker.embedding(f) for f in frames[: min(10, len(frames))]]
    reference = aggregate_embeddings(ref_embeddings)
    if reference is None:
        raise RuntimeError("Attacker could not obtain a reference face embedding")

    rows = []
    for alpha in alphas:
        protected = [apply_transform(method, f, float(alpha)) for f in frames]
        embeddings = [attacker.embedding(f) for f in protected]
        valid = [e for e in embeddings if e is not None]
        curve = cumulative_sequence_curve(reference, embeddings)
        finite_curve = [float(x) for x in curve if np.isfinite(x)]
        frame_scores = [float(np.dot(reference, e)) for e in valid]
        rows.append({
            "attacker": attacker_name,
            "method": method,
            "alpha": float(alpha),
            "sampled_frames": len(frames),
            "recognized_frames": len(valid),
            "frame_similarity_mean": float(np.mean(frame_scores)) if frame_scores else None,
            "frame_similarity_max": float(np.max(frame_scores)) if frame_scores else None,
            "sequence_similarity_final": finite_curve[-1] if finite_curve else None,
            "sequence_similarity_max": max(finite_curve) if finite_curve else None,
            "cumulative_curve": curve,
        })

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    print("NOTE: empirical similarity under the selected attacker; not a formal privacy guarantee.")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--attacker", choices=["insightface", "facenet"], default="insightface")
    p.add_argument("--method", choices=["original", "blur", "pixelation", "static_tapf"], default="static_tapf")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--output", default="results/sequence_privacy.json")
    args = p.parse_args()
    run(args.video, args.attacker, args.method, stride=args.stride, max_frames=args.max_frames, output=args.output)
