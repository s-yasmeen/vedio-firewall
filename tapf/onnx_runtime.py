"""Optional ONNX Runtime adapter for edge inference.

The deployment service itself does not ingest raw biometrics. This adapter is intended
for the patient-device/edge component where task and privacy models execute locally.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time


@dataclass(frozen=True)
class RuntimeInfo:
    available: bool
    providers: tuple[str, ...]
    model_path: str | None = None


class OnnxEdgeModel:
    def __init__(self, model_path: str, providers=None):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(model_path)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install requirements-edge.txt to enable ONNX Runtime") from exc
        wanted = providers or ["CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        selected = [p for p in wanted if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=selected)
        self.model_path = str(path)
        self.providers = tuple(self.session.get_providers())

    def run(self, feeds: dict):
        start = time.perf_counter()
        outputs = self.session.run(None, feeds)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return outputs, latency_ms

    @property
    def info(self):
        return RuntimeInfo(True, self.providers, self.model_path)


def runtime_info() -> RuntimeInfo:
    try:
        import onnxruntime as ort
        return RuntimeInfo(True, tuple(ort.get_available_providers()))
    except Exception:
        return RuntimeInfo(False, tuple())
