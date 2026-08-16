"""Observed outcome monitoring and human-readable monitoring report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from churn_platform.models.evaluate import evaluate_predictions


def evaluate_observed_performance(
    current: pd.DataFrame,
    budget_fraction: float,
) -> dict[str, Any] | None:
    """Evaluate new labeled outcomes when both labels and probabilities are available."""
    if not {"churn", "churn_probability"}.issubset(current.columns):
        return None
    labeled = current.dropna(subset=["churn", "churn_probability"])
    if labeled.empty or labeled["churn"].nunique() < 2:
        return None
    return evaluate_predictions(
        labeled["churn"], labeled["churn_probability"].to_numpy(), budget_fraction
    )


def render_monitoring_report(
    drift: dict[str, Any],
    performance: dict[str, Any] | None,
    destination: str | Path,
) -> None:
    """Render a reproducible Markdown monitoring report with action ownership."""
    alert_rows = drift["alerts"]
    lines = [
        "# Monitoring Report",
        "",
        "This report compares the scored batch with the training reference. Alerts are diagnostic; "
        "they do not trigger automatic retraining or customer actions.",
        "",
        "## Batch status",
        "",
        f"- Status: **{drift['status'].upper()}**",
        f"- Reference rows: {drift['baseline_rows']}",
        f"- Current rows: {drift['current_rows']}",
        f"- Alerts: {len(alert_rows)}",
        "",
        "## Alerts",
        "",
    ]
    if alert_rows:
        lines.extend(["| Severity | Column | Issue | Value |", "|---|---|---|---|"])
        for alert in alert_rows:
            lines.append(
                f"| {alert['severity']} | {alert['column']} | {alert['issue']} | "
                f"{alert.get('value', '')} |"
            )
    else:
        lines.append("No thresholds were exceeded.")
    lines.extend(["", "## Observed performance", ""])
    if performance:
        for key in (
            "roc_auc",
            "average_precision",
            "brier_score",
            "recall_at_budget",
            "precision_at_budget",
            "lift_at_budget",
        ):
            lines.append(f"- {key}: {performance[key]:.4f}")
    else:
        lines.append("No mature labels were supplied; performance drift cannot yet be assessed.")
    lines.extend(
        [
            "",
            "## Operating guidance",
            "",
            "Missing columns and numeric type violations should block scoring. Range, missingness, "
            "PSI, and KS alerts require a data owner and model owner to investigate context before "
            "retraining, changing thresholds, or stopping a campaign. Any decline in labeled "
            "performance requires human review of label maturity, segment mix, calibration, and "
            "business costs.",
        ]
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
