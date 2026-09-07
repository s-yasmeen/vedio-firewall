"""CREMA-D video transform runner for TAPF-MIN.

Creates transformed video clips and a per-clip experiment manifest. It intentionally
separates representation generation from identity/task model evaluation so the same
protected clips are scored by multiple independent attackers.
"""
from pathlib import Path
import argparse, csv
import cv2
from tapf.transforms import apply_transform
from datasets.cremad_manifest import build_manifest

METHODS = {
    "original": 0.0,
    "blur": 0.85,
    "pixelation": 0.85,
    "static_tapf": 0.65,
}


def transform_video(src, dst, method, alpha, max_frames=None):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(apply_transform(method, frame, alpha))
        count += 1
        if max_frames and count >= max_frames:
            break
    cap.release()
    writer.release()
    return count


def run(root, output="results/cremad_protected", limit=0, max_frames=None):
    root = Path(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / "source_manifest.csv"
    rows = build_manifest(str(root), str(manifest_file))
    if limit:
        rows = rows[:limit]

    generated = []
    for row in rows:
        src = Path(row["path"])
        for method, alpha in METHODS.items():
            dst = output / row["split"] / method / f"{src.stem}.mp4"
            n = transform_video(src, dst, method, alpha, max_frames=max_frames)
            generated.append({**row, "method": method, "alpha": alpha,
                              "protected_path": str(dst), "frames": n})

    out_manifest = output / "protected_manifest.csv"
    if generated:
        with open(out_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=generated[0].keys())
            writer.writeheader(); writer.writerows(generated)
    print(f"Generated {len(generated)} transformed clip records -> {out_manifest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="CREMA-D video root")
    ap.add_argument("--output", default="results/cremad_protected")
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()
    run(args.root, args.output, args.limit, args.max_frames)
