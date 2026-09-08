"""TAPF-MIN v3 raw-video FER + formal output-privacy benchmark on CREMA-D.

This experiment intentionally reports three different layers rather than collapsing them:

A. LOCAL FER UTILITY
   MobileNetV3-Small + causal GRU + temporal attention, evaluated on actors never seen
   by the fold's model.

B. EMPIRICAL IDENTITY LEAKAGE
   Independent post-hoc attackers are trained on the exact non-private task posterior.
   This measures residual biometric linkage but is not a formal theorem.

C. FORMAL RELEASE UTILITY
   The local FER decision is randomized with k-ary randomized response. Each released
   label is epsilon-local-DP. Utility is measured on the randomized labels themselves.

The experiment does not claim clinical validity. Formal DP applies to the released task
label under the categorical-input adjacency definition; it does not imply anonymity or
protect side channels outside the mechanism.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import random
import re

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tapf.fer_v3 import FERV3Config, TAPFMinFERV3
from tapf.formal_privacy import randomized_response
from experiments.run_cremad_final_validation import _attack, repeated_release_attack
from experiments.run_cremad_privacy_utility_spectrum import bootstrap_metric


NAME_RE = re.compile(r"^(\d{4})_DFA_(ANG|DIS|FEA|HAP|NEU|SAD)_XX\.flv$", re.I)
EMOTIONS = np.asarray(["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"])
EMO_TO_I = {e: i for i, e in enumerate(EMOTIONS)}
MIN_REAL_VIDEO_BYTES = 1000
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_uniform_frames(path: str, frames_per_clip: int = 12, size: int = 112) -> torch.Tensor:
    """Read approximately uniform RGB frames from one CREMA-D clip."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count < 2:
        cap.release()
        raise RuntimeError(f"Insufficient video frames: {path}")
    wanted = np.linspace(0, count - 1, num=frames_per_clip).round().astype(int)
    out = []
    for idx in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed reading frame {idx}: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        out.append(torch.from_numpy(x).permute(2, 0, 1))
    cap.release()
    return torch.stack(out, dim=0)


class CremadVideoDataset(Dataset):
    def __init__(self, records, *, actor_to_i=None, frames_per_clip=12, image_size=112, augment=False, seed=42):
        self.records = list(records)
        self.actor_to_i = actor_to_i
        self.frames_per_clip = int(frames_per_clip)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.seed = int(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, actor, emotion, original_idx = self.records[idx]
        x = read_uniform_frames(str(path), self.frames_per_clip, self.image_size)
        if self.augment:
            # Deterministic per-sample flip keeps experiments reproducible.
            rng = np.random.default_rng(self.seed + int(original_idx))
            if rng.random() < 0.5:
                x = torch.flip(x, dims=[3])
        y = EMO_TO_I[emotion]
        identity = -1 if self.actor_to_i is None else self.actor_to_i[actor]
        return x, torch.tensor(y, dtype=torch.long), torch.tensor(identity, dtype=torch.long), int(original_idx)


def _class_weights(records, device):
    counts = np.zeros(len(EMOTIONS), dtype=np.float64)
    for _, _, emo, _ in records:
        counts[EMO_TO_I[emo]] += 1
    weights = counts.sum() / np.maximum(1.0, len(EMOTIONS) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_fold(
    train_records,
    test_records,
    *,
    epochs: int,
    batch_size: int,
    frames_per_clip: int,
    image_size: int,
    seed: int,
    pretrained: bool,
    device: torch.device,
):
    actors = sorted({a for _, a, _, _ in train_records})
    actor_to_i = {a: i for i, a in enumerate(actors)}
    train_ds = CremadVideoDataset(
        train_records, actor_to_i=actor_to_i, frames_per_clip=frames_per_clip,
        image_size=image_size, augment=True, seed=seed,
    )
    test_ds = CremadVideoDataset(
        test_records, actor_to_i=None, frames_per_clip=frames_per_clip,
        image_size=image_size, augment=False, seed=seed,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    config = FERV3Config(
        n_emotions=len(EMOTIONS), n_identities=len(actors), temporal_hidden=192,
        temporal_layers=2, dropout=0.25, pretrained=pretrained,
    )
    model = TAPFMinFERV3(config).to(device)
    class_weights = _class_weights(train_records, device)

    # Stage 1 stabilizes the temporal/task heads. Stage 2 fine-tunes the final backbone blocks.
    freeze_epochs = max(1, min(2, epochs // 3)) if epochs > 1 else 0
    if freeze_epochs:
        model.freeze_backbone()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-3, weight_decay=1e-4)

    history = []
    for epoch in range(epochs):
        if epoch == freeze_epochs and freeze_epochs:
            model.unfreeze_backbone_tail(3)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-4, weight_decay=1e-4)
        model.train()
        losses = []
        # Smoothly ramp the identity adversary to avoid destabilizing early FER learning.
        progress = (epoch + 1) / max(1, epochs)
        grl_strength = float(2.0 / (1.0 + np.exp(-6.0 * progress)) - 1.0)
        for frames, emo, identity, _ in train_loader:
            frames = frames.to(device); emo = emo.to(device); identity = identity.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(frames, identity_grl_strength=grl_strength)
            emotion_loss = F.cross_entropy(out["emotion_logits"], emo, weight=class_weights, label_smoothing=0.05)
            identity_loss = F.cross_entropy(out["identity_logits"], identity)
            loss = emotion_loss + 0.20 * identity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "grl_strength": grl_strength})

    model.eval()
    posterior_rows, indices = [], []
    with torch.no_grad():
        for frames, _, _, original_idx in test_loader:
            p = model.task_posterior(frames.to(device)).cpu().numpy()
            posterior_rows.append(p)
            indices.extend(np.asarray(original_idx).astype(int).tolist())
    return np.vstack(posterior_rows), np.asarray(indices, dtype=int), history, asdict(config)


def _dp_label_sweep(local_pred, true_emotion, epsilons, seed=42, monte_carlo=100):
    rows = []
    for e_i, epsilon in enumerate(epsilons):
        # Exact expected top-1 accuracy under k-ary RR conditioned on the local prediction.
        k = len(EMOTIONS)
        exp_eps = np.exp(float(epsilon))
        p_true = exp_eps / (exp_eps + k - 1)
        p_other = 1.0 / (exp_eps + k - 1)
        local_correct = (local_pred == true_emotion)
        expected_accuracy = float(np.mean(np.where(local_correct, p_true, p_other)))

        f1_values = []
        acc_values = []
        for rep in range(int(monte_carlo)):
            rng = np.random.default_rng(seed + e_i * 10000 + rep)
            released = np.asarray([
                randomized_response(str(label), EMOTIONS.tolist(), epsilon=float(epsilon), rng=rng)["label"]
                for label in local_pred
            ])
            f1_values.append(float(f1_score(true_emotion, released, average="macro", zero_division=0)))
            acc_values.append(float(accuracy_score(true_emotion, released)))
        rows.append({
            "mechanism": "kary_randomized_response",
            "epsilon_per_release": float(epsilon),
            "delta_per_release": 0.0,
            "formal_guarantee": "epsilon-local-DP per released categorical FER label",
            "expected_accuracy_exact": expected_accuracy,
            "monte_carlo_accuracy_mean": float(np.mean(acc_values)),
            "monte_carlo_macro_f1_mean": float(np.mean(f1_values)),
            "monte_carlo_macro_f1_ci95": [float(np.quantile(f1_values, .025)), float(np.quantile(f1_values, .975))],
            "monte_carlo_repetitions": int(monte_carlo),
        })
    return rows


def run(
    root,
    *,
    folds=5,
    epochs=8,
    batch_size=8,
    frames_per_clip=12,
    image_size=112,
    seed=42,
    pretrained=True,
):
    seed_all(seed)
    root = Path(root)
    raw = []
    for p in sorted(root.glob("*.flv")):
        m = NAME_RE.match(p.name)
        if m and p.stat().st_size > MIN_REAL_VIDEO_BYTES:
            raw.append((p, m.group(1), m.group(2).upper()))
    actor = np.asarray([a for _, a, _ in raw])
    emotion = np.asarray([e for _, _, e in raw])
    if len(raw) < 120 or len(np.unique(actor)) < 8:
        raise RuntimeError(f"Need >=120 clips and >=8 actors; got {len(raw)} clips/{len(np.unique(actor))} actors")

    records = [(p, a, e, i) for i, (p, a, e) in enumerate(raw)]
    folds = max(2, min(int(folds), len(np.unique(actor))))
    splitter = GroupKFold(n_splits=folds)
    P = np.zeros((len(records), len(EMOTIONS)), dtype=np.float32)
    fold_rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (tr, te) in enumerate(splitter.split(np.arange(len(records)), emotion, groups=actor), 1):
        train_records = [records[i] for i in tr]
        test_records = [records[i] for i in te]
        Pf, ids, history, config = train_one_fold(
            train_records, test_records, epochs=epochs, batch_size=batch_size,
            frames_per_clip=frames_per_clip, image_size=image_size,
            seed=seed + fold * 101, pretrained=pretrained, device=device,
        )
        P[ids] = Pf
        pred = EMOTIONS[np.argmax(Pf, axis=1)]
        fold_rows.append({
            "fold": fold,
            "train_actors": int(len(np.unique(actor[tr]))),
            "test_actors": int(len(np.unique(actor[te]))),
            "actor_overlap": int(len(set(actor[tr]).intersection(set(actor[te])))),
            "macro_f1": float(f1_score(emotion[te], pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(emotion[te], pred)),
            "training_history": history,
            "model_config": config,
        })

    local_pred = EMOTIONS[np.argmax(P, axis=1)]
    f1, f1_lo, f1_hi = bootstrap_metric(
        emotion, local_pred, lambda y, p: f1_score(y, p, average="macro", zero_division=0),
        seed=seed + 700, n=500,
    )
    acc = float(accuracy_score(emotion, local_pred))
    posterior_privacy = _attack(P, actor, seed + 900, bootstrap_n=400)
    repeated = repeated_release_attack(P, actor, seed + 1200)
    dp_sweep = _dp_label_sweep(local_pred, emotion, [0.25, 0.5, 1.0, 2.0, 4.0], seed=seed + 1500)

    result = {
        "dataset": "CREMA-D DFA cohort supplied to runner",
        "clips": len(records),
        "actors": int(len(np.unique(actor))),
        "model": "TAPF-MIN v3 MobileNetV3-Small + causal GRU + temporal attention + GRL identity adversary",
        "device": str(device),
        "scientific_scope": "Raw-video controlled FER benchmark; not clinical validation.",
        "local_fer": {
            "accuracy": acc,
            "macro_f1": float(f1),
            "macro_f1_ci95": [float(f1_lo), float(f1_hi)],
            "folds": fold_rows,
            "actor_disjoint": bool(all(r["actor_overlap"] == 0 for r in fold_rows)),
        },
        "empirical_privacy_of_nonprivate_posterior": {
            "clip_identity": posterior_privacy,
            "repeated_release_identity": repeated,
            "interpretation": "Empirical attacker evidence only; not a formal guarantee.",
        },
        "formal_private_release": {
            "released_object": "randomized categorical FER label only",
            "mechanism": "k-ary randomized response",
            "adjacency": "any two possible private categorical FER inputs",
            "composition": "Basic sequential composition: N releases consume N*epsilon total epsilon at delta=0.",
            "utility_sweep": dp_sweep,
        },
        "minimum_disclosure_note": "The private label is smaller than posterior or latent release. Posterior/latent release requires a separately calibrated DP mechanism and joint empirical gate.",
        "training_parameters": {
            "folds": folds, "epochs": epochs, "batch_size": batch_size,
            "frames_per_clip": frames_per_clip, "image_size": image_size,
            "pretrained_imagenet": bool(pretrained), "seed": seed,
        },
    }
    Path("results").mkdir(exist_ok=True)
    out = Path("results/cremad_v3_deep_fer_formal_privacy.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--frames-per-clip", type=int, default=12)
    ap.add_argument("--image-size", type=int, default=112)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()
    run(
        args.root, folds=args.folds, epochs=args.epochs, batch_size=args.batch_size,
        frames_per_clip=args.frames_per_clip, image_size=args.image_size,
        seed=args.seed, pretrained=not args.no_pretrained,
    )
