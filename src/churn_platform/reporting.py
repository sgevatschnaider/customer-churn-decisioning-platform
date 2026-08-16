"""Reproducible model-card and business-report rendering from run artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from churn_platform.decisioning.economics import EconomicScenario
from churn_platform.models.train import ModelBundle


def _metric_list(metrics: dict[str, Any]) -> list[str]:
    return [
        f"- ROC-AUC: {metrics['roc_auc']:.4f}",
        f"- PR-AUC / average precision: {metrics['average_precision']:.4f}",
        f"- Brier score: {metrics['brier_score']:.4f}",
        f"- Precision at budget: {metrics['precision_at_budget']:.4f}",
        f"- Recall at budget: {metrics['recall_at_budget']:.4f}",
        f"- Lift at budget: {metrics['lift_at_budget']:.4f}",
        f"- F1 at budget: {metrics['f1']:.4f}",
    ]


def render_model_card(
    bundle: ModelBundle,
    snapshots: pd.DataFrame,
    source_name: str,
    destination: str | Path,
) -> None:
    """Render a model card grounded in the serialized test metrics."""
    feature_start = snapshots["cutoff_date"].min().date()
    feature_end = snapshots["cutoff_date"].max().date()
    lines = [
        "# Model Card",
        "",
        "## Purpose",
        "",
        "Prioritize customers for retention review using calibrated churn risk and a separate "
        "economic scenario layer. The system is production-oriented, not a claim of live "
        "enterprise deployment.",
        "",
        "## Population and data",
        "",
        f"- Data source: {source_name}",
        f"- Snapshot rows: {len(snapshots):,}",
        f"- Distinct source customer identifiers: {snapshots['customer_id'].nunique():,}",
        f"- Feature cutoffs: {feature_start} to {feature_end}",
        "- Target: no positive purchase in the 45 days strictly after a snapshot cutoff.",
        "- Features: recency, frequency, monetary value, order value, tenure, invoice/product "
        "counts, regularity, recent trends, returns, quantity, geography, and window-over-window "
        "changes.",
        "",
        "## Validation strategy",
        "",
        "Models were fitted on historical snapshots, selected only on a later validation cutoff, "
        "calibrated on that validation period, and evaluated once on the final temporal test "
        "cutoff. "
        "Label windows end before the next partition cutoff.",
        "",
        "## Selected model",
        "",
        f"- Model: {bundle.model_name}",
        f"- Version: `{bundle.model_version}`",
        f"- Trained at: {bundle.trained_at_utc}",
        "",
        "## Test metrics",
        "",
        *_metric_list(bundle.test_metrics),
        "",
        "The operating confusion matrix is stored in `artifacts/metrics.json`; its positive class "
        "is the budget-constrained contact policy, not a universal 0.5 classification threshold.",
        "",
        "## Limitations and risks",
        "",
        "The source has no retention treatment, campaign response, marketing consent, customer "
        "acquisition cost, or causal outcome. The configured retention probability is an "
        "assumption, "
        "not an identified treatment effect. Expected net value is therefore scenario analysis and "
        "must not be presented as guaranteed incremental profit.",
        "",
        "Customer IDs are operational pseudonyms, not identities. Geographic segment may proxy for "
        "protected or commercially sensitive attributes; it requires fairness review and "
        "lawful-use assessment before deployment. The model should not be used for pricing, "
        "credit, eligibility, "
        "or adverse customer treatment.",
        "",
        "## Conditions for non-use",
        "",
        "Do not use with incomplete horizons, schema failures, unresolved drift alerts, a "
        "materially different customer population, unreviewed campaign costs, or where outreach "
        "lacks a lawful basis.",
        "",
        "## Monitoring",
        "",
        "Validate schema and drift for every scoring batch, review probability distributions "
        "weekly, and assess calibration and ranking performance after labels mature. Human owners "
        "decide whether "
        "alerts justify pausing, investigation, or retraining.",
    ]
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_business_report(
    comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    scenario: EconomicScenario,
    source_name: str,
    destination: str | Path,
) -> None:
    """Render decision-policy results and economic caveats from actual run outputs."""
    value = comparison.set_index("policy").loc["expected_value"]
    table_columns = [
        "policy",
        "customers_contacted",
        "recall_at_budget",
        "precision_at_budget",
        "lift_at_budget",
        "expected_net_value",
        "scenario_realized_net_value",
    ]
    table = comparison[table_columns].copy()
    for column in ("recall_at_budget", "precision_at_budget", "lift_at_budget"):
        table[column] = table[column].map(lambda value: f"{value:.3f}")
    for column in ("expected_net_value", "scenario_realized_net_value"):
        table[column] = table[column].map(lambda value: f"{value:,.2f}")
    lines = [
        "# Business Results",
        "",
        "## Executive summary",
        "",
        f"This report is generated from an actual **{source_name}** pipeline run. Under a "
        f"{scenario.max_contact_fraction:.0%} contact limit, the value-aware policy selected "
        f"{int(value['customers_contacted']):,} customers, captured "
        f"{value['recall_at_budget']:.1%} of observed churners, and produced scenario expected net "
        f"value of {scenario.currency} {value['expected_net_value']:,.2f}.",
        "",
        "## Budget and recommended policy",
        "",
        f"- Scenario budget: {scenario.currency} {scenario.total_budget:,.2f}",
        f"- Contact cost: {scenario.currency} {scenario.contact_cost:,.2f}",
        f"- Offer cost: {scenario.currency} {scenario.offer_cost:,.2f}",
        f"- Assumed retention probability: {scenario.retention_probability:.0%}",
        f"- Maximum contact fraction: {scenario.max_contact_fraction:.0%}",
        "- Recommendation: use positive expected net value ranking as a review queue, subject to "
        "consent, operational capacity, and an experimental campaign design.",
        "",
        "## Policy comparison",
        "",
        table.to_markdown(index=False),
        "",
        "The random benchmark reports the mean across 200 deterministic seeded policy draws. "
        "`scenario_realized_net_value` uses observed churn labels and the same assumed retention "
        "rate; it is still not causal profit.",
        "",
        "## Sensitivity",
        "",
        f"Across {len(sensitivity)} unique configurations, expected net value ranged from "
        f"{scenario.currency} {sensitivity['expected_net_value'].min():,.2f} to "
        f"{scenario.currency} {sensitivity['expected_net_value'].max():,.2f}. The analysis varies "
        "retention probability, offer cost, and gross-margin scaling.",
        "",
        "## Risks and next steps",
        "",
        "The source contains no treatment assignment or campaign response, so retention "
        "probability is a configurable scenario input rather than an estimated causal effect. A "
        "real next step is a "
        "randomized controlled retention experiment, followed by uplift modeling and segment-level "
        "fairness, consent, deliverability, and capacity checks.",
    ]
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
