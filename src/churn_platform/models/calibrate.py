"""Temporal holdout probability calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


@dataclass
class ProbabilityCalibrator(ClassifierMixin, BaseEstimator):
    """Wrap a fitted classifier with sigmoid calibration learned on a later holdout."""

    base_estimator: Any
    calibrator: LogisticRegression

    @property
    def classes_(self) -> np.ndarray:
        """Expose the conventional fitted classifier class order."""
        return np.array([0, 1])

    @classmethod
    def fit(
        cls,
        base_estimator: Any,
        validation_features: pd.DataFrame,
        validation_target: pd.Series,
    ) -> ProbabilityCalibrator:
        """Fit calibration only on the temporal validation split."""
        raw_probability = base_estimator.predict_proba(validation_features)[:, 1]
        calibrator = LogisticRegression(random_state=42)
        calibrator.fit(_logit(raw_probability), validation_target)
        return cls(base_estimator=base_estimator, calibrator=calibrator)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return calibrated two-class probabilities."""
        raw_probability = self.base_estimator.predict_proba(features)[:, 1]
        calibrated = self.calibrator.predict_proba(_logit(raw_probability))[:, 1]
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return a conventional 0.5-threshold prediction."""
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)
