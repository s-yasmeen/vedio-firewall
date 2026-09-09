"""TAPF-MIN v3.1 motion-aware FER repair.

This model addresses four weaknesses in v3:
1. face-focused inputs rather than whole frames,
2. explicit temporal motion-residual encoding,
3. gated RGB/motion fusion before temporal modeling,
4. delayed, tunable gradient-reversal identity suppression.

The model is still a research FER model. Formal privacy remains at the release layer.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from tapf.fer_v3 import TemporalAttention, gradient_reverse


@dataclass(frozen=True)
class FERV31Config:
    n_emotions: int = 6
    n_identities: int = 24
    temporal_hidden: int = 192
    temporal_layers: int = 2
    motion_dim: int = 96
    dropout: float = 0.25
    pretrained: bool = True
    identity_weight: float = 0.10
    grl_start_fraction: float = 0.35
    grl_max_strength: float = 0.60


class MotionResidualEncoder(nn.Module):
    """Compact encoder over adjacent normalized-frame differences."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(96, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, residuals):
        return self.proj(self.net(residuals))


class GatedFusion(nn.Module):
    def __init__(self, rgb_dim: int, motion_dim: int, out_dim: int):
        super().__init__()
        self.rgb_proj = nn.Linear(rgb_dim, out_dim)
        self.motion_proj = nn.Linear(motion_dim, out_dim)
        self.gate = nn.Sequential(nn.Linear(rgb_dim + motion_dim, out_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, rgb, motion):
        gate = self.gate(torch.cat([rgb, motion], dim=-1))
        fused = gate * self.rgb_proj(rgb) + (1.0 - gate) * self.motion_proj(motion)
        return self.norm(fused), gate


class TAPFMinFERV31(nn.Module):
    """Face-focused RGB + motion residual + temporal attention FER."""

    def __init__(self, config: FERV31Config | None = None):
        super().__init__()
        self.config = config or FERV31Config()
        weights = MobileNet_V3_Small_Weights.DEFAULT if self.config.pretrained else None
        base = mobilenet_v3_small(weights=weights)
        self.rgb_encoder = base.features
        rgb_dim = int(base.classifier[0].in_features)
        self.rgb_pool = nn.AdaptiveAvgPool2d(1)
        self.rgb_norm = nn.LayerNorm(rgb_dim)
        self.motion_encoder = MotionResidualEncoder(self.config.motion_dim)
        self.fusion = GatedFusion(rgb_dim, self.config.motion_dim, self.config.temporal_hidden)
        self.temporal = nn.GRU(
            self.config.temporal_hidden, self.config.temporal_hidden,
            num_layers=self.config.temporal_layers,
            dropout=self.config.dropout if self.config.temporal_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attn = TemporalAttention(self.config.temporal_hidden)
        self.task_projection = nn.Sequential(
            nn.LayerNorm(self.config.temporal_hidden),
            nn.Linear(self.config.temporal_hidden, self.config.temporal_hidden),
            nn.GELU(), nn.Dropout(self.config.dropout),
        )
        self.emotion_head = nn.Linear(self.config.temporal_hidden, self.config.n_emotions)
        self.identity_head = nn.Sequential(
            nn.Linear(self.config.temporal_hidden, 128), nn.GELU(), nn.Dropout(self.config.dropout),
            nn.Linear(128, self.config.n_identities),
        )

    @staticmethod
    def scheduled_grl(progress: float, config: FERV31Config) -> float:
        """Delay identity suppression until FER has learned a useful task representation."""
        p = float(min(1.0, max(0.0, progress)))
        start = float(config.grl_start_fraction)
        if p <= start:
            return 0.0
        q = (p - start) / max(1e-8, 1.0 - start)
        smooth = 2.0 / (1.0 + math.exp(-6.0 * q)) - 1.0
        return float(config.grl_max_strength * smooth)

    def encode_rgb(self, frames):
        b, t, c, h, w = frames.shape
        x = frames.reshape(b*t, c, h, w)
        x = self.rgb_encoder(x)
        x = self.rgb_pool(x).flatten(1)
        return self.rgb_norm(x).reshape(b, t, -1)

    def encode_motion(self, frames):
        # First frame has zero residual to preserve sequence length.
        residual = torch.zeros_like(frames)
        residual[:, 1:] = frames[:, 1:] - frames[:, :-1]
        b, t, c, h, w = residual.shape
        z = self.motion_encoder(residual.reshape(b*t, c, h, w))
        return z.reshape(b, t, -1)

    def forward(self, frames, *, identity_grl_strength: float = 0.0, mask=None):
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError('frames must have shape [B,T,3,H,W]')
        rgb = self.encode_rgb(frames)
        motion = self.encode_motion(frames)
        fused, gate = self.fusion(rgb, motion)
        seq, _ = self.temporal(fused)
        pooled, attn = self.attn(seq, mask=mask)
        latent = self.task_projection(pooled)
        return {
            'emotion_logits': self.emotion_head(latent),
            'identity_logits': self.identity_head(gradient_reverse(latent, identity_grl_strength)),
            'task_latent': latent,
            'temporal_attention': attn,
            'fusion_gate': gate,
        }

    @torch.no_grad()
    def task_posterior(self, frames, mask=None):
        out = self.forward(frames, identity_grl_strength=0.0, mask=mask)
        return torch.softmax(out['emotion_logits'], dim=-1)

    def freeze_rgb_backbone(self):
        for p in self.rgb_encoder.parameters(): p.requires_grad = False

    def unfreeze_rgb_tail(self, blocks: int = 4):
        for module in list(self.rgb_encoder.children())[-max(1, int(blocks)):]:
            for p in module.parameters(): p.requires_grad = True


def fer_v31_loss(outputs, emotion_targets, identity_targets, *, identity_weight: float):
    emo = F.cross_entropy(outputs['emotion_logits'], emotion_targets, label_smoothing=0.05)
    ident = F.cross_entropy(outputs['identity_logits'], identity_targets)
    return {'loss': emo + float(identity_weight)*ident, 'emotion_loss': emo, 'identity_adversary_loss': ident}
