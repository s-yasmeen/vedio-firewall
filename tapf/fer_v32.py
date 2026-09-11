"""TAPF-MIN FER v3.2: utility-first visual FER model.

The release/privacy architecture remains unchanged.  This module focuses on fixing the
current dominant weakness: task utility.  It uses a stronger ImageNet-pretrained
EfficientNet-B0 frame encoder, explicit motion residuals, gated fusion, and a temporal
GRU/attention head.  Identity suppression is optional and can be delayed until after a
utility-capable FER representation has been learned.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from tapf.fer_v3 import TemporalAttention, gradient_reverse


@dataclass(frozen=True)
class FERV32Config:
    n_emotions: int = 6
    n_identities: int = 91
    temporal_hidden: int = 256
    temporal_layers: int = 2
    motion_dim: int = 128
    dropout: float = 0.30
    pretrained: bool = True
    identity_weight: float = 0.05
    grl_start_fraction: float = 0.60
    grl_max_strength: float = 0.35


class MotionEncoderV32(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 160, 3, 2, 1, bias=False), nn.BatchNorm2d(160), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(160, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class FusionV32(nn.Module):
    def __init__(self, rgb_dim: int, motion_dim: int, hidden: int):
        super().__init__()
        self.rgb_proj = nn.Sequential(nn.Linear(rgb_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.motion_proj = nn.Sequential(nn.Linear(motion_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(rgb_dim + motion_dim, hidden), nn.Sigmoid())
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, rgb, motion):
        g = self.gate(torch.cat([rgb, motion], dim=-1))
        z = g * self.rgb_proj(rgb) + (1.0 - g) * self.motion_proj(motion)
        return self.out_norm(z), g


class TAPFMinFERV32(nn.Module):
    def __init__(self, config: FERV32Config | None = None):
        super().__init__()
        self.config = config or FERV32Config()
        weights = EfficientNet_B0_Weights.DEFAULT if self.config.pretrained else None
        base = efficientnet_b0(weights=weights)
        self.rgb_encoder = base.features
        rgb_dim = int(base.classifier[1].in_features)
        self.rgb_pool = nn.AdaptiveAvgPool2d(1)
        self.rgb_norm = nn.LayerNorm(rgb_dim)
        self.motion_encoder = MotionEncoderV32(self.config.motion_dim)
        self.fusion = FusionV32(rgb_dim, self.config.motion_dim, self.config.temporal_hidden)
        self.temporal = nn.GRU(
            self.config.temporal_hidden,
            self.config.temporal_hidden,
            num_layers=self.config.temporal_layers,
            dropout=self.config.dropout if self.config.temporal_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
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
    def scheduled_grl(progress: float, config: FERV32Config) -> float:
        p = float(min(1.0, max(0.0, progress)))
        if p <= config.grl_start_fraction:
            return 0.0
        q = (p - config.grl_start_fraction) / max(1e-8, 1.0 - config.grl_start_fraction)
        return float(config.grl_max_strength * (2.0 / (1.0 + math.exp(-6.0*q)) - 1.0))

    def encode_rgb(self, frames):
        b,t,c,h,w = frames.shape
        x = self.rgb_encoder(frames.reshape(b*t,c,h,w))
        x = self.rgb_pool(x).flatten(1)
        return self.rgb_norm(x).reshape(b,t,-1)

    def encode_motion(self, frames):
        residual = torch.zeros_like(frames)
        residual[:,1:] = frames[:,1:] - frames[:,:-1]
        b,t,c,h,w = residual.shape
        z = self.motion_encoder(residual.reshape(b*t,c,h,w))
        return z.reshape(b,t,-1)

    def forward(self, frames, *, identity_grl_strength: float = 0.0, mask=None):
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        rgb = self.encode_rgb(frames)
        motion = self.encode_motion(frames)
        fused, gate = self.fusion(rgb, motion)
        seq,_ = self.temporal(fused)
        pooled, attn = self.attn(seq, mask=mask)
        latent = self.task_projection(pooled)
        return {
            "emotion_logits": self.emotion_head(latent),
            "identity_logits": self.identity_head(gradient_reverse(latent, identity_grl_strength)),
            "task_latent": latent,
            "temporal_attention": attn,
            "fusion_gate": gate,
        }

    def freeze_rgb_backbone(self):
        for p in self.rgb_encoder.parameters(): p.requires_grad = False

    def unfreeze_rgb_tail(self, blocks: int = 3):
        for module in list(self.rgb_encoder.children())[-max(1, int(blocks)):]:
            for p in module.parameters(): p.requires_grad = True
