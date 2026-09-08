"""TAPF-MIN v3 raw-video facial-expression model.

The model is deliberately separate from the formal privacy mechanism. Its job is to
maximize authorized FER utility and suppress identity features empirically. Formal
privacy is applied only to the released task output by ``tapf.formal_privacy``.

Architecture
------------
- MobileNetV3-Small frame encoder (ImageNet weights optional).
- Causal temporal GRU over frame embeddings.
- Learned temporal attention pooling.
- FER task head.
- Gradient-reversal identity adversary used only during training.
- Only the FER posterior/label is intended for the release layer; the latent embedding
  is not an authorized release representation.

This is a research model, not a clinical diagnostic model.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
except ImportError as exc:  # pragma: no cover - exercised only without research deps
    raise RuntimeError(
        "TAPF-MIN v3 FER requires torch and torchvision; install requirements-research.txt"
    ) from exc


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength: float):
        ctx.strength = float(strength)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


def gradient_reverse(x, strength: float):
    return _GradientReverse.apply(x, strength)


@dataclass(frozen=True)
class FERV3Config:
    n_emotions: int = 6
    n_identities: int = 24
    temporal_hidden: int = 192
    temporal_layers: int = 2
    dropout: float = 0.25
    pretrained: bool = False


class TemporalAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, max(32, dim // 2)),
            nn.Tanh(),
            nn.Linear(max(32, dim // 2), 1),
        )

    def forward(self, sequence, mask=None):
        # sequence: [B,T,D]
        logits = self.score(sequence).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), float("-inf"))
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class TAPFMinFERV3(nn.Module):
    """MobileNetV3 + temporal GRU FER with training-time identity adversary."""

    def __init__(self, config: FERV3Config | None = None):
        super().__init__()
        self.config = config or FERV3Config()
        if self.config.n_emotions < 2:
            raise ValueError("n_emotions must be >= 2")
        if self.config.n_identities < 2:
            raise ValueError("n_identities must be >= 2")

        weights = MobileNet_V3_Small_Weights.DEFAULT if self.config.pretrained else None
        base = mobilenet_v3_small(weights=weights)
        self.frame_encoder = base.features
        frame_dim = int(base.classifier[0].in_features)
        self.frame_pool = nn.AdaptiveAvgPool2d(1)
        self.frame_norm = nn.LayerNorm(frame_dim)

        self.temporal = nn.GRU(
            input_size=frame_dim,
            hidden_size=self.config.temporal_hidden,
            num_layers=self.config.temporal_layers,
            dropout=self.config.dropout if self.config.temporal_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.temporal_attention = TemporalAttention(self.config.temporal_hidden)
        self.task_projection = nn.Sequential(
            nn.LayerNorm(self.config.temporal_hidden),
            nn.Linear(self.config.temporal_hidden, self.config.temporal_hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.emotion_head = nn.Linear(self.config.temporal_hidden, self.config.n_emotions)
        self.identity_adversary = nn.Sequential(
            nn.Linear(self.config.temporal_hidden, 128),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, self.config.n_identities),
        )

    def encode_frames(self, frames):
        """Encode [B,T,3,H,W] normalized RGB frames to [B,T,D]."""
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w)
        x = self.frame_encoder(x)
        x = self.frame_pool(x).flatten(1)
        x = self.frame_norm(x)
        return x.reshape(b, t, -1)

    def forward(self, frames, *, identity_grl_strength: float = 0.0, mask=None):
        frame_embeddings = self.encode_frames(frames)
        temporal, _ = self.temporal(frame_embeddings)
        pooled, attention = self.temporal_attention(temporal, mask=mask)
        task_latent = self.task_projection(pooled)
        emotion_logits = self.emotion_head(task_latent)
        identity_logits = self.identity_adversary(
            gradient_reverse(task_latent, identity_grl_strength)
        )
        return {
            "emotion_logits": emotion_logits,
            "identity_logits": identity_logits,
            "task_latent": task_latent,
            "temporal_attention": attention,
        }

    @torch.no_grad()
    def task_posterior(self, frames, mask=None):
        """Return the local FER posterior; callers must not release it without policy."""
        out = self.forward(frames, identity_grl_strength=0.0, mask=mask)
        return torch.softmax(out["emotion_logits"], dim=-1)

    def freeze_backbone(self):
        for p in self.frame_encoder.parameters():
            p.requires_grad = False

    def unfreeze_backbone_tail(self, blocks: int = 3):
        blocks = max(1, int(blocks))
        children = list(self.frame_encoder.children())
        for module in children[-blocks:]:
            for p in module.parameters():
                p.requires_grad = True


def multitask_loss(outputs, emotion_targets, identity_targets, *, identity_weight: float = 0.20):
    """FER loss plus GRL adversary loss.

    The GRL is applied in the model, so minimizing identity CE improves the adversary
    while reversing its gradient at the task representation.
    """
    task = F.cross_entropy(outputs["emotion_logits"], emotion_targets)
    identity = F.cross_entropy(outputs["identity_logits"], identity_targets)
    total = task + float(identity_weight) * identity
    return {"loss": total, "emotion_loss": task, "identity_adversary_loss": identity}
