"""Batch inference with model-contract validation."""

from __future__ import annotations

import pandas as pd

from churn_platform.models.train import ModelBundle


def score_customers(bundle: ModelBundle, features: pd.DataFrame) -> pd.DataFrame:
    """Score customer features and retain their input row order."""
    missing = sorted(set(bundle.feature_names) - set(features.columns))
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    scored = features.copy()
    scored["churn_probability"] = bundle.estimator.predict_proba(scored[bundle.feature_names])[:, 1]
    scored["model_version"] = bundle.model_version
    return scored
