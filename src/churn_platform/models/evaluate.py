"""Predictive, calibration, and budget-aware evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def budget_selection(probability: np.ndarray, fraction: float) -> np.ndarray:
    """Select the top probabilities under an exact customer-contact fraction."""
    if not 0 < fraction <= 1:
        raise ValueError("budget fraction must be in (0, 1]")
    count = max(1, int(np.floor(len(probability) * fraction)))
    selected_index = np.argsort(-probability, kind="stable")[:count]
    selected = np.zeros(len(probability), dtype=bool)
    selected[selected_index] = True
    return selected


def evaluate_predictions(
    target: pd.Series | np.ndarray,
    probability: np.ndarray,
    budget_fraction: float,
) -> dict[str, Any]:
    """Compute ranking, calibration, and operating-policy metrics."""
    y_true = np.asarray(target, dtype=int)
    y_probability = np.asarray(probability, dtype=float)
    if len(y_true) != len(y_probability):
        raise ValueError("target and probability lengths differ")
    selected = budget_selection(y_probability, budget_fraction)
    policy_prediction = selected.astype(int)
    positives = int(y_true.sum())
    base_rate = float(y_true.mean())
    recall_at_budget = float(y_true[selected].sum() / positives) if positives else 0.0
    precision_at_budget = float(y_true[selected].mean()) if selected.any() else 0.0
    lift_at_budget = precision_at_budget / base_rate if base_rate else 0.0
    matrix = confusion_matrix(y_true, policy_prediction, labels=[0, 1])
    return {
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "average_precision": float(average_precision_score(y_true, y_probability)),
        "precision": float(precision_score(y_true, policy_prediction, zero_division=0)),
        "recall": float(recall_score(y_true, policy_prediction, zero_division=0)),
        "f1": float(f1_score(y_true, policy_prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "recall_at_budget": recall_at_budget,
        "precision_at_budget": precision_at_budget,
        "lift_at_budget": float(lift_at_budget),
        "budget_fraction": float(budget_fraction),
        "customers": int(len(y_true)),
        "contacted": int(selected.sum()),
        "observed_churners": positives,
        "confusion_matrix": matrix.tolist(),
    }


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_evaluation_plots(
    target: pd.Series | np.ndarray,
    probability: np.ndarray,
    output_directory: str | Path,
    feature_importance: pd.Series | None = None,
) -> list[Path]:
    """Generate calibration, precision-recall, gains, and importance plots."""
    output = Path(output_directory)
    y_true = np.asarray(target, dtype=int)
    y_probability = np.asarray(probability, dtype=float)
    paths: list[Path] = []

    observed, predicted = calibration_curve(y_true, y_probability, n_bins=8, strategy="quantile")
    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(predicted, observed, marker="o", label="Selected model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed churn rate")
    plt.title("Probability calibration")
    plt.legend()
    path = output / "calibration_curve.png"
    _save_figure(path)
    paths.append(path)

    precision, recall, _ = precision_recall_curve(y_true, y_probability)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color="#006D77")
    plt.axhline(y_true.mean(), linestyle="--", color="gray", label="Base rate")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    plt.legend()
    path = output / "precision_recall_curve.png"
    _save_figure(path)
    paths.append(path)

    order = np.argsort(-y_probability, kind="stable")
    cumulative = np.cumsum(y_true[order])
    population = np.arange(1, len(y_true) + 1) / len(y_true)
    gains = cumulative / max(cumulative[-1], 1)
    plt.figure(figsize=(6, 4))
    plt.plot(population, gains, label="Selected model", color="#E76F51")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("Fraction of customers contacted")
    plt.ylabel("Fraction of churners captured")
    plt.title("Cumulative gains")
    plt.legend()
    path = output / "gains_chart.png"
    _save_figure(path)
    paths.append(path)

    if feature_importance is not None and not feature_importance.empty:
        importance = feature_importance.sort_values().tail(12)
        plt.figure(figsize=(7, 5))
        plt.barh(importance.index, importance.values, color="#457B9D")
        plt.xlabel("Decrease in average precision")
        plt.title("Permutation importance on the test period")
        path = output / "feature_importance.png"
        _save_figure(path)
        paths.append(path)
    return paths
