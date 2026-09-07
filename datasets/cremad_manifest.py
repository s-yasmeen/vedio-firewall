"""Build deterministic subject-disjoint CREMA-D manifests from AudioWAV or VideoFlash files."""
from pathlib import Path
import csv, hashlib

EMOTIONS = {"ANG":"anger", "DIS":"disgust", "FEA":"fear", "HAP":"happy", "NEU":"neutral", "SAD":"sad"}


def stable_bucket(actor_id: str) -> int:
    return int(hashlib.sha256(actor_id.encode()).hexdigest()[:8], 16) % 100


def split_actor(actor_id: str) -> str:
    bucket = stable_bucket(actor_id)
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def parse_clip(path: Path):
    # Typical CREMA-D filename: ActorID_Sentence_Emotion_Level.ext
    parts = path.stem.split("_")
    if len(parts) < 4 or parts[2] not in EMOTIONS:
        return None
    actor = parts[0]
    return {
        "path": str(path),
        "actor_id": actor,
        "emotion": EMOTIONS[parts[2]],
        "split": split_actor(actor),
    }


def build_manifest(root: str, output_csv: str):
    root = Path(root)
    extensions = {".flv", ".mp4", ".mov", ".avi"}
    rows = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            row = parse_clip(p)
            if row:
                rows.append(row)
    rows.sort(key=lambda x: x["path"])
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "actor_id", "emotion", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--output", default="cremad_manifest.csv")
    args = ap.parse_args()
    rows = build_manifest(args.root, args.output)
    print(f"Wrote {len(rows)} video records to {args.output}")
