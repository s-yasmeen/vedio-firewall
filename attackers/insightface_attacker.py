"""Optional InsightFace/ArcFace identity attacker adapter.

This module is intentionally optional and is not imported by the core package or CI.
InsightFace pretrained model licensing/usage terms must be reviewed before use.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class InsightFaceConfig:
    model_name: str = "buffalo_l"
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    ctx_id: int = -1


class InsightFaceAttacker:
    """ArcFace-family embedding attacker using InsightFace FaceAnalysis."""

    def __init__(self, config: InsightFaceConfig | None = None):
        self.config = config or InsightFaceConfig()
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "InsightFace is optional. Install the research extra: "
                "pip install insightface onnxruntime"
            ) from exc
        self.app = FaceAnalysis(name=self.config.model_name, providers=list(self.config.providers))
        self.app.prepare(ctx_id=self.config.ctx_id)

    def embedding(self, bgr_frame: np.ndarray):
        faces = self.app.get(bgr_frame)
        if not faces:
            return None
        face = max(faces, key=lambda f: float(f.bbox[2] - f.bbox[0]) * float(f.bbox[3] - f.bbox[1]))
        emb = np.asarray(face.embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    @staticmethod
    def cosine_similarity(a, b) -> float:
        if a is None or b is None:
            return float("nan")
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
