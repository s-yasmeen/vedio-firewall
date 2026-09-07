"""Optional FaceNet attacker using facenet-pytorch.

Not imported by the core package or CI. Model weights are downloaded by facenet-pytorch
when a pretrained model is requested; record the exact model/version used in experiments.
"""
from __future__ import annotations
import numpy as np


class FaceNetAttacker:
    def __init__(self, device: str = "cpu", pretrained: str = "vggface2"):
        try:
            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1
        except ImportError as exc:
            raise RuntimeError(
                "FaceNet is optional. Install with: pip install facenet-pytorch torch torchvision"
            ) from exc
        self.torch = torch
        self.device = device
        self.mtcnn = MTCNN(image_size=160, margin=0, device=device)
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(device)

    def embedding(self, bgr_frame: np.ndarray):
        from PIL import Image
        rgb = bgr_frame[:, :, ::-1]
        face = self.mtcnn(Image.fromarray(rgb))
        if face is None:
            return None
        with self.torch.no_grad():
            emb = self.model(face.unsqueeze(0).to(self.device)).cpu().numpy()[0]
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    @staticmethod
    def cosine_similarity(a, b) -> float:
        if a is None or b is None:
            return float("nan")
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
