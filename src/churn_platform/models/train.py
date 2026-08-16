"""Temporal model comparison, selection, calibration, and experiment tracking."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_platform.config import PROJECT_ROOT, project_path
from churn_platform.features.build_features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)
from churn_platform.models.calibrate import ProbabilityCalibrator
from churn_platform.models.evaluate import evaluate_predictions, generate_evaluation_plots

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Serializable inference contract and model lineage."""

    estimator: ProbabilityCalibrator
    model_name: str
    model_version: str
    trained_at_utc: str
    feature_names: list[str]
    training_period: dict[str, str]
    test_metrics: dict[str, Any]
    validation_comparison: list[dict[str, Any]]


def _preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def _candidate_models(config: dict[str, Any]) -> list[tuple[str, Pipeline]]:
    seed = int(config.get("random_seed", 42))
    candidates: list[tuple[str, Pipeline]] = []
    for c_value in config.get("logistic_c_values", [1.0]):
        candidates.append(
            (
                f"logistic_regression_c_{c_value:g}",
                Pipeline(
                    [
                        ("preprocessor", _preprocessor(scale=True)),
                        (
                            "classifier",
                            LogisticRegression(
                                C=float(c_value),
                                class_weight="balanced",
                                max_iter=1_000,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            )
        )
    hgb = config.get("hist_gradient_boosting", {})
    candidates.append(
        (
            "hist_gradient_boosting",
            Pipeline(
                [
                    ("preprocessor", _preprocessor(scale=False)),
                    (
                        "classifier",
                        HistGradientBoostingClassifier(
                            learning_rate=float(hgb.get("learning_rate", 0.08)),
                            max_iter=int(hgb.get("max_iter", 180)),
                            max_leaf_nodes=int(hgb.get("max_leaf_nodes", 15)),
                            l2_regularization=float(hgb.get("l2_regularization", 1.0)),
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        )
    )
    return candidates


def _heuristic_probability(features: pd.DataFrame) -> np.ndarray:
    recency_rank = features["recency_days"].rank(pct=True).to_numpy()
    trend_rank = (-features["recent_purchasing_trend"]).rank(pct=True).to_numpy()
    return np.clip(0.7 * recency_rank + 0.3 * trend_rank, 0.001, 0.999)


def code_version() -> str:
    """Return the Git commit SHA when available, otherwise an explicit local marker."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted-local-run"


def train_and_select(
    snapshots: pd.DataFrame,
    model_config: dict[str, Any],
    model_path: str | Path = "artifacts/model.joblib",
    figures_directory: str | Path = "reports/figures",
    tracking_uri: str | None = None,
    decisioning_config: dict[str, Any] | None = None,
) -> tuple[ModelBundle, pd.DataFrame]:
    """Select on validation, calibrate on validation, and evaluate test exactly once."""
    required_splits = {"train", "validation", "test"}
    if not required_splits.issubset(set(snapshots["split"])):
        raise ValueError("Snapshots must contain train, validation, and test splits")
    budget_fraction = float(model_config.get("budget_fraction", 0.15))
    train = snapshots.loc[snapshots["split"].eq("train")]
    validation = snapshots.loc[snapshots["split"].eq("validation")]
    test = snapshots.loc[snapshots["split"].eq("test")]
    x_train, y_train = train[MODEL_FEATURES].copy(), train["churn"]
    x_validation, y_validation = validation[MODEL_FEATURES].copy(), validation["churn"]
    x_test, y_test = test[MODEL_FEATURES].copy(), test["churn"]
    for frame in (x_train, x_validation, x_test):
        frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(float)

    comparison: list[dict[str, Any]] = []
    heuristic_metrics = evaluate_predictions(
        y_validation, _heuristic_probability(x_validation), budget_fraction
    )
    comparison.append({"model": "recency_trend_heuristic", **heuristic_metrics})

    fitted_candidates: dict[str, Pipeline] = {}
    for name, candidate in _candidate_models(model_config):
        candidate.fit(x_train, y_train)
        probability = candidate.predict_proba(x_validation)[:, 1]
        metrics = evaluate_predictions(y_validation, probability, budget_fraction)
        comparison.append({"model": name, **metrics})
        fitted_candidates[name] = candidate
        LOGGER.info("Validation %s average_precision=%.4f", name, metrics["average_precision"])

    machine_candidates = [row for row in comparison if row["model"] in fitted_candidates]
    winner = max(
        machine_candidates,
        key=lambda row: (row["average_precision"], -row["brier_score"]),
    )
    selected_base = fitted_candidates[str(winner["model"])]
    calibrated = ProbabilityCalibrator.fit(selected_base, x_validation, y_validation)
    test_probability = calibrated.predict_proba(x_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_probability, budget_fraction)

    importance_result = permutation_importance(
        calibrated,
        x_test,
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=int(model_config.get("random_seed", 42)),
        n_jobs=1,
    )
    feature_importance = pd.Series(importance_result.importances_mean, index=MODEL_FEATURES)
    plot_paths = generate_evaluation_plots(
        y_test, test_probability, project_path(figures_directory), feature_importance
    )

    bundle = ModelBundle(
        estimator=calibrated,
        model_name=str(winner["model"]),
        model_version=code_version(),
        trained_at_utc=datetime.now(UTC).isoformat(),
        feature_names=list(MODEL_FEATURES),
        training_period={
            "feature_cutoff_start": train["cutoff_date"].min().isoformat(),
            "feature_cutoff_end": train["cutoff_date"].max().isoformat(),
            "validation_cutoff": validation["cutoff_date"].min().isoformat(),
            "test_cutoff": test["cutoff_date"].min().isoformat(),
        },
        test_metrics=test_metrics,
        validation_comparison=comparison,
    )
    destination = project_path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)

    scored_test = test.copy()
    scored_test["churn_probability"] = test_probability
    scored_test["model_version"] = bundle.model_version
    metrics_path = destination.parent / "metrics.json"
    metrics_path.write_text(
        json.dumps({"test": test_metrics, "validation": comparison}, indent=2),
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(tracking_uri or project_path("mlruns").as_uri())
    mlflow.set_experiment("customer-churn-decisioning")
    with mlflow.start_run(run_name=f"{bundle.model_name}-temporal-test") as run:
        mlflow.log_params(
            {
                "selected_model": bundle.model_name,
                "features": ",".join(MODEL_FEATURES),
                "train_start": bundle.training_period["feature_cutoff_start"],
                "train_end": bundle.training_period["feature_cutoff_end"],
                "validation_cutoff": bundle.training_period["validation_cutoff"],
                "test_cutoff": bundle.training_period["test_cutoff"],
                "code_version": bundle.model_version,
            }
        )
        mlflow.log_metrics(
            {key: value for key, value in test_metrics.items() if isinstance(value, float | int)}
        )
        if decisioning_config:
            mlflow.log_dict(decisioning_config, "configuration/decisioning.json")
        mlflow.log_dict(model_config, "configuration/model.json")
        mlflow.log_dict({"features": MODEL_FEATURES}, "model/features.json")
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        for plot_path in plot_paths:
            mlflow.log_artifact(str(plot_path), artifact_path="plots")
        mlflow.sklearn.log_model(
            calibrated,
            artifact_path="model",
            code_paths=[str(project_path("src/churn_platform"))],
            input_example=x_test.head(3),
            signature=infer_signature(x_test, test_probability),
            pip_requirements=[
                "cloudpickle==3.1.2",
                "numpy==2.1.3",
                "pandas==2.2.3",
                "scikit-learn==1.6.0",
            ],
        )
        (destination.parent / "mlflow_run.json").write_text(
            json.dumps(
                {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id}, indent=2
            ),
            encoding="utf-8",
        )
    return bundle, scored_test


def load_model_bundle(path: str | Path = "artifacts/model.joblib") -> ModelBundle:
    """Load and validate the serialized model contract."""
    bundle = joblib.load(project_path(path))
    if not isinstance(bundle, ModelBundle):
        raise TypeError("Serialized artifact is not a ModelBundle")
    return bundle
