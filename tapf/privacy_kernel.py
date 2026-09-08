"""TAPF-MIN v3 minimum-disclosure formal privacy kernel.

The kernel is intended to run at the trusted edge after local task inference. Raw video
and task latents are outside its release API. It selects the smallest pre-declared task
output whose frozen evidence satisfies utility + empirical privacy constraints and whose
formal privacy spend fits the remaining budget, then applies the randomized mechanism.

Minimum disclosure is an auditable engineering property over a declared representation
lattice (measured payload bits). Differential privacy is the formal privacy property.
They are reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from tapf.formal_privacy import (
    BasicDPAccountant,
    DisclosureCandidate,
    FormalPrivacyBudget,
    gaussian_release,
    randomized_response,
    select_minimum_disclosure,
)


@dataclass(frozen=True)
class ReleaseEvidence:
    name: str
    mechanism: str
    utility_lower: float
    empirical_identity_risk_upper: float
    epsilon: float
    delta: float = 0.0
    quantization_bits: int | None = None


class MinimumDisclosurePrivacyKernel:
    """Stateful edge privacy-budget + release controller.

    The accountant state must be persisted by the deployment layer across process restarts
    for a production guarantee spanning multiple releases. This in-memory object is the
    mechanism core and is suitable for controlled experiments/tests.
    """

    def __init__(
        self,
        classes: Sequence[str],
        *,
        epsilon_budget: float,
        delta_budget: float = 0.0,
        posterior_clip_norm: float = 1.0,
    ):
        labels = tuple(str(x) for x in classes)
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("classes must contain at least two unique labels")
        self.classes = labels
        self.posterior_clip_norm = float(posterior_clip_norm)
        self.accountant = BasicDPAccountant(FormalPrivacyBudget(epsilon_budget, delta_budget))

    @property
    def label_payload_bits(self) -> int:
        return int(math.ceil(math.log2(len(self.classes))))

    def _candidate(self, evidence: ReleaseEvidence) -> DisclosureCandidate:
        if evidence.mechanism == "randomized_response_label":
            bits = self.label_payload_bits
        elif evidence.mechanism == "gaussian_quantized_posterior":
            q = int(evidence.quantization_bits or 8)
            if q < 1 or q > 16:
                raise ValueError("posterior quantization_bits must be in [1,16]")
            bits = len(self.classes) * q
        else:
            raise ValueError(f"Unsupported mechanism: {evidence.mechanism}")
        return DisclosureCandidate(
            name=evidence.name,
            disclosure_bits=bits,
            utility_lower=evidence.utility_lower,
            empirical_identity_risk_upper=evidence.empirical_identity_risk_upper,
            formal_epsilon=evidence.epsilon,
            formal_delta=evidence.delta,
        )

    @staticmethod
    def _project_simplex(x: np.ndarray) -> np.ndarray:
        """Euclidean projection to the probability simplex (DP-safe post-processing)."""
        v = np.asarray(x, dtype=np.float64).reshape(-1)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(1, len(v) + 1)
        cond = u - cssv / ind > 0
        if not np.any(cond):
            return np.full_like(v, 1.0 / len(v))
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
        return np.maximum(v - theta, 0.0)

    @staticmethod
    def _quantize_probability_vector(p: np.ndarray, bits: int) -> np.ndarray:
        levels = (1 << int(bits)) - 1
        q = np.round(np.clip(p, 0.0, 1.0) * levels) / levels
        total = float(q.sum())
        if total <= 0:
            return np.full_like(q, 1.0 / len(q))
        return q / total

    def release(
        self,
        local_posterior: Sequence[float],
        evidence: Sequence[ReleaseEvidence],
        *,
        min_utility_lower: float,
        max_empirical_identity_risk_upper: float,
        rng: np.random.Generator | None = None,
    ) -> dict:
        posterior = np.asarray(local_posterior, dtype=np.float64).reshape(-1)
        if posterior.size != len(self.classes) or not np.isfinite(posterior).all():
            raise ValueError("local_posterior must be finite and match class count")
        if np.any(posterior < 0) or float(posterior.sum()) <= 0:
            raise ValueError("local_posterior must contain non-negative mass")
        posterior = posterior / posterior.sum()

        evidence_by_name = {e.name: e for e in evidence}
        if len(evidence_by_name) != len(evidence):
            raise ValueError("release evidence names must be unique")
        candidates = [self._candidate(e) for e in evidence]
        selection = select_minimum_disclosure(
            candidates,
            min_utility_lower=min_utility_lower,
            max_empirical_identity_risk_upper=max_empirical_identity_risk_upper,
            accountant=self.accountant,
        )
        if selection["decision"] != "RELEASE":
            return {
                **selection,
                "raw_video_released": False,
                "latent_released": False,
                "formal_privacy": self.accountant.statement(),
            }

        chosen = evidence_by_name[selection["selected"]]
        generator = rng or np.random.default_rng()
        if chosen.mechanism == "randomized_response_label":
            true_label = self.classes[int(np.argmax(posterior))]
            private = randomized_response(
                true_label,
                self.classes,
                epsilon=chosen.epsilon,
                rng=generator,
                accountant=self.accountant,
            )
            payload = {"type": "categorical_label", "value": private["label"]}
            mechanism = private
        else:
            private = gaussian_release(
                posterior,
                epsilon=chosen.epsilon,
                delta=chosen.delta,
                clip_norm=self.posterior_clip_norm,
                rng=generator,
                accountant=self.accountant,
            )
            p = self._project_simplex(private["values"])
            qbits = int(chosen.quantization_bits or 8)
            p = self._quantize_probability_vector(p, qbits)
            payload = {"type": "quantized_posterior", "values": p.astype(np.float32).tolist(), "quantization_bits": qbits}
            mechanism = {**private, "values": None, "post_processing": "simplex_projection_then_quantization"}

        return {
            **selection,
            "payload": payload,
            "mechanism": mechanism,
            "raw_video_released": False,
            "latent_released": False,
            "formal_privacy": self.accountant.statement(),
            "scope": "Formal DP applies to the released task object under the mechanism adjacency definition; empirical identity evidence and minimum-disclosure size are separate properties.",
        }
