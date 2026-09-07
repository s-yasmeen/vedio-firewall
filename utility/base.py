"""Downstream task utility interfaces for TAPF-MIN."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np


class TaskUtilityEvaluator(ABC):
    name: str = "task-utility"

    @abstractmethod
    def utility(self, frames: Sequence[np.ndarray], target=None) -> float:
        """Return task utility in [0,1]; larger means better task preservation."""
        raise NotImplementedError


class ClassificationConfidenceUtility(TaskUtilityEvaluator):
    """Generic adapter for a classifier returning class probabilities."""

    def __init__(self, predictor, name="classification-confidence"):
        self.predictor = predictor
        self.name = name

    def utility(self, frames: Sequence[np.ndarray], target=None) -> float:
        if target is None:
            raise ValueError("target class is required for classification utility")
        values = []
        for frame in frames:
            probs = np.asarray(self.predictor(frame), dtype=float).reshape(-1)
            if target >= len(probs):
                raise ValueError("target class out of range")
            values.append(float(probs[target]))
        return float(np.mean(values)) if values else 0.0
