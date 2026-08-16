"""Reference-based schema, quality, PSI, KS, and prediction drift detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from churn_platform.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate PSI from reference quantile bins with numerical stabilization."""
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(reference_values) == 0 or len(current_values) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.allclose(np.mean(reference_values), np.mean(current_values)) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts = np.histogram(reference_values, bins=edges)[0] / len(reference_values)
    current_counts = np.histogram(current_values, bins=edges)[0] / len(current_values)
    reference_counts = np.clip(reference_counts, 1e-6, None)
    current_counts = np.clip(current_counts, 1e-6, None)
    return float(
        np.sum((current_counts - reference_counts) * np.log(current_counts / reference_counts))
    )


def build_monitoring_baseline(reference: pd.DataFrame) -> dict[str, Any]:
    """Capture schema and reference distributions needed by online/offline monitoring."""
    columns: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        if column in reference:
            values = pd.to_numeric(reference[column], errors="coerce")
            columns[column] = {
                "kind": "numeric",
                "dtype": str(reference[column].dtype),
                "missing_rate": float(reference[column].isna().mean()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "quantiles": [
                    float(value)
                    for value in values.quantile(np.linspace(0, 1, 11)).dropna().unique()
                ],
                "sample": [float(value) for value in values.dropna().head(2_000)],
            }
    for column in CATEGORICAL_FEATURES:
        if column in reference:
            columns[column] = {
                "kind": "categorical",
                "dtype": str(reference[column].dtype),
                "missing_rate": float(reference[column].isna().mean()),
                "categories": sorted(reference[column].dropna().astype(str).unique().tolist()),
            }
    if "churn_probability" in reference:
        values = reference["churn_probability"].dropna()
        columns["churn_probability"] = {
            "kind": "probability",
            "dtype": str(reference["churn_probability"].dtype),
            "missing_rate": float(reference["churn_probability"].isna().mean()),
            "minimum": 0.0,
            "maximum": 1.0,
            "sample": [float(value) for value in values.head(2_000)],
        }
    return {"row_count": len(reference), "columns": columns}


def compare_to_baseline(
    baseline: dict[str, Any],
    current: pd.DataFrame,
    psi_warning: float = 0.20,
    ks_warning: float = 0.15,
    missing_increase_warning: float = 0.05,
) -> dict[str, Any]:
    """Compare a new batch to baseline and return diagnostic alerts."""
    expected = baseline["columns"]
    missing_columns = sorted(set(expected) - set(current.columns))
    alerts: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    for column in missing_columns:
        alerts.append({"severity": "blocking", "column": column, "issue": "missing_column"})

    for column, specification in expected.items():
        if column not in current:
            continue
        values = current[column]
        missing_rate = float(values.isna().mean())
        column_metrics: dict[str, Any] = {"missing_rate": missing_rate}
        if missing_rate - float(specification["missing_rate"]) > missing_increase_warning:
            alerts.append(
                {
                    "severity": "warning",
                    "column": column,
                    "issue": "missing_rate_increase",
                    "value": missing_rate,
                }
            )
        kind = specification["kind"]
        if kind in {"numeric", "probability"}:
            numeric = pd.to_numeric(values, errors="coerce")
            type_error_rate = float((numeric.isna() & values.notna()).mean())
            column_metrics["type_error_rate"] = type_error_rate
            if type_error_rate > 0:
                alerts.append(
                    {
                        "severity": "blocking",
                        "column": column,
                        "issue": "invalid_numeric_type",
                        "value": type_error_rate,
                    }
                )
            outside = numeric.lt(specification["minimum"]) | numeric.gt(specification["maximum"])
            outside_rate = float(outside.fillna(False).mean())
            column_metrics["outside_reference_range_rate"] = outside_rate
            if outside_rate > 0.01:
                alerts.append(
                    {
                        "severity": "warning",
                        "column": column,
                        "issue": "values_outside_reference_range",
                        "value": outside_rate,
                    }
                )
            reference_sample = pd.Series(specification.get("sample", []), dtype=float)
            psi = population_stability_index(reference_sample, numeric)
            ks = (
                float(ks_2samp(reference_sample, numeric.dropna()).statistic)
                if len(reference_sample) and numeric.notna().any()
                else float("nan")
            )
            column_metrics.update({"psi": psi, "ks_statistic": ks})
            if np.isfinite(psi) and psi >= psi_warning:
                alerts.append(
                    {"severity": "warning", "column": column, "issue": "psi_drift", "value": psi}
                )
            if np.isfinite(ks) and ks >= ks_warning:
                alerts.append(
                    {"severity": "warning", "column": column, "issue": "ks_drift", "value": ks}
                )
        else:
            new_categories = sorted(
                set(values.dropna().astype(str)) - set(specification.get("categories", []))
            )
            column_metrics["new_categories"] = new_categories
            if new_categories:
                alerts.append(
                    {
                        "severity": "warning",
                        "column": column,
                        "issue": "new_categories",
                        "value": new_categories,
                    }
                )
        metrics[column] = column_metrics
    return {
        "baseline_rows": baseline["row_count"],
        "current_rows": len(current),
        "missing_columns": missing_columns,
        "metrics": metrics,
        "alerts": alerts,
        "status": "alert" if alerts else "ok",
    }


def save_baseline(baseline: dict[str, Any], path: str | Path) -> None:
    """Persist a monitoring baseline."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(baseline, indent=2), encoding="utf-8")


def load_baseline(path: str | Path) -> dict[str, Any]:
    """Load a monitoring baseline."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
