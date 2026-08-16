from __future__ import annotations

import numpy as np
import pandas as pd

from churn_platform.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from churn_platform.monitoring.data_drift import (
    build_monitoring_baseline,
    compare_to_baseline,
    load_baseline,
    population_stability_index,
    save_baseline,
)
from churn_platform.monitoring.performance import (
    evaluate_observed_performance,
    render_monitoring_report,
)


def monitoring_frame(rows: int = 100) -> pd.DataFrame:
    frame = pd.DataFrame({column: np.linspace(1, 10, rows) for column in NUMERIC_FEATURES})
    frame[CATEGORICAL_FEATURES[0]] = "Europe"
    frame["churn_probability"] = np.linspace(0.01, 0.99, rows)
    frame["churn"] = np.tile([0, 1], rows // 2)
    return frame


def test_baseline_and_drift_alerts(tmp_path) -> None:
    reference = monitoring_frame()
    baseline = build_monitoring_baseline(reference)
    path = tmp_path / "baseline.json"
    save_baseline(baseline, path)
    current = reference.copy()
    current = current.drop(columns=[NUMERIC_FEATURES[0]])
    current.loc[:20, NUMERIC_FEATURES[1]] = np.nan
    current[CATEGORICAL_FEATURES[0]] = "New segment"
    result = compare_to_baseline(load_baseline(path), current)
    issues = {alert["issue"] for alert in result["alerts"]}
    assert result["status"] == "alert"
    assert "missing_column" in issues
    assert "missing_rate_increase" in issues
    assert "new_categories" in issues


def test_psi_performance_and_report(tmp_path) -> None:
    reference = pd.Series(np.linspace(0, 1, 100))
    assert population_stability_index(reference, reference) == 0
    frame = monitoring_frame()
    performance = evaluate_observed_performance(frame, 0.15)
    assert performance is not None
    drift = compare_to_baseline(build_monitoring_baseline(frame), frame)
    report = tmp_path / "monitoring.md"
    render_monitoring_report(drift, performance, report)
    assert "human review" in report.read_text(encoding="utf-8")
    assert evaluate_observed_performance(frame.drop(columns="churn"), 0.15) is None
