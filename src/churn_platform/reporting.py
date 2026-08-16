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
    eligibility_report: dict[str, Any] | None = None,
) -> None:
    """Render a model card grounded in the serialized test metrics."""
    feature_start = snapshots["cutoff_date"].min().date()
    feature_end = snapshots["cutoff_date"].max().date()
    metadata = bundle.run_metadata
    eligibility = (eligibility_report or {}).get("overall", {})
    test_eligibility = next(
        (
            item
            for item in (eligibility_report or {}).get("by_cutoff", [])
            if item.get("split") == "test"
        ),
        {},
    )
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
        f"- Customers before eligibility across cutoffs: "
        f"{eligibility.get('total_customers_before_eligibility', 'not recorded')}",
        f"- Eligible active-repeat observations: "
        f"{eligibility.get('eligible_customers', len(snapshots))}",
        f"- Excluded observations: {eligibility.get('excluded_customers', 'not recorded')}",
        f"- Final test customers before eligibility: "
        f"{test_eligibility.get('total_customers_before_eligibility', 'not recorded')}",
        f"- Final eligible test customers: "
        f"{test_eligibility.get('eligible_customers', 'not recorded')}",
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
        f"- Exact public source commit: `{metadata.get('source_commit', 'not recorded')}`",
        f"- Execution timestamp: " f"{metadata.get('execution_timestamp_utc', 'not recorded')}",
        f"- Trained at: {bundle.trained_at_utc}",
        f"- MLflow tracking URI: `{bundle.tracking_uri}`",
        f"- Dataset SHA-256: `{metadata.get('dataset_sha256', 'not recorded')}`",
        f"- Dependency lock identifier: "
        f"`{metadata.get('dependency_lock_identifier', 'not recorded')}`",
        f"- Python: {metadata.get('python_version', 'not recorded')}",
        f"- Configuration hashes: `{metadata.get('configuration_hashes', {})}`",
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
        "acquisition cost, or causal outcome. The incremental-retention effect and offer "
        "acceptance probability are separate scenario assumptions, "
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
    eligibility_report: dict[str, Any],
) -> None:
    """Render decision-policy results and economic caveats from actual run outputs."""
    value = comparison.set_index("policy").loc["expected_value"]
    churn_policy = comparison.set_index("policy").loc["churn_probability"]
    test_eligibility = next(
        item for item in eligibility_report["by_cutoff"] if item["split"] == "test"
    )
    if value["lift_at_budget"] < 1:
        lift_explanation = (
            f"Its observed lift is {value['lift_at_budget']:.3f}, below 1; this result is reported "
            "without suppression because the policy optimizes scenario value, not churn capture."
        )
    else:
        lift_explanation = (
            f"Its observed lift is {value['lift_at_budget']:.3f}, above random selection but lower "
            f"than the churn-probability policy at {churn_policy['lift_at_budget']:.3f}."
        )
    table_columns = [
        "policy",
        "customers_contacted",
        "average_churn_probability",
        "average_estimated_margin_at_risk",
        "expected_net_value_per_contact",
        "observed_churn_rate",
        "recall_at_budget",
        "precision_at_budget",
        "lift_at_budget",
        "expected_net_value",
        "expected_campaign_cost",
        "binding_constraint",
        "scenario_realized_net_value",
    ]
    table = comparison[table_columns].copy()
    for column in (
        "average_churn_probability",
        "observed_churn_rate",
        "recall_at_budget",
        "precision_at_budget",
        "lift_at_budget",
    ):
        table[column] = table[column].map(lambda value: f"{value:.3f}")
    for column in (
        "average_estimated_margin_at_risk",
        "expected_net_value_per_contact",
        "expected_net_value",
        "expected_campaign_cost",
        "scenario_realized_net_value",
    ):
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
        "## Operationally eligible population",
        "",
        f"- Customer observations before eligibility: "
        f"{eligibility_report['overall']['total_customers_before_eligibility']:,}",
        f"- Eligible active-repeat observations: "
        f"{eligibility_report['overall']['eligible_customers']:,}",
        f"- Excluded observations: {eligibility_report['overall']['excluded_customers']:,}",
        f"- Final test customers before eligibility: "
        f"{test_eligibility['total_customers_before_eligibility']:,}",
        f"- Final eligible test customers: {test_eligibility['eligible_customers']:,}",
        f"- Final test exclusions: {test_eligibility['excluded_customers']:,}",
        f"- Eligible-population churn prevalence: "
        f"{eligibility_report['overall']['eligible_churn_prevalence']:.1%}",
        f"- Exclusion criteria failures: "
        f"`{eligibility_report['overall']['criterion_failure_counts']}`",
        f"- Primary exclusion reasons: "
        f"`{eligibility_report['overall']['primary_exclusion_reasons']}`",
        "",
        "## Budget and recommended policy",
        "",
        f"- Scenario budget: {scenario.currency} {scenario.total_budget:,.2f}",
        f"- Contact cost: {scenario.currency} {scenario.contact_cost:,.2f}",
        f"- Offer cost if accepted: {scenario.currency} " f"{scenario.offer_cost_if_accepted:,.2f}",
        f"- Offer acceptance probability: {scenario.offer_acceptance_probability:.0%}",
        f"- Incremental retention effect: {scenario.incremental_retention_effect:.0%}",
        f"- Economic and prediction horizon: {scenario.economic_horizon_days} days",
        f"- Maximum contact fraction: {scenario.max_contact_fraction:.0%}",
        f"- Budget-based contact capacity: {int(value['budget_based_contact_capacity']):,}",
        f"- Operations-based contact capacity: "
        f"{int(value['operations_based_contact_capacity']):,}",
        f"- Economically eligible customers: {int(value['economically_eligible_customers']):,}",
        f"- Actual selected customers: {int(value['actual_selected_customers']):,}",
        f"- Binding constraint: `{value['binding_constraint']}`",
        f"- Expected campaign cost: {scenario.currency} " f"{value['expected_campaign_cost']:,.2f}",
        f"- Remaining budget: {scenario.currency} {value['remaining_budget']:,.2f}",
        f"- Budget utilization: {value['budget_utilization_percentage']:.1f}%",
        f"- Expected value per contacted customer: {scenario.currency} "
        f"{value['expected_net_value_per_contact']:,.2f}",
        "- Recommendation: use positive expected net value ranking as a review queue, subject to "
        "consent, operational capacity, and an experimental campaign design.",
        "",
        "## Policy comparison",
        "",
        table.to_markdown(index=False),
        "",
        "The random benchmark reports the mean across 200 deterministic seeded policy draws. "
        "`scenario_realized_net_value` uses observed churn labels and the configured incremental "
        "effect; it is still not causal profit.",
        "",
        "## Value-versus-lift decomposition",
        "",
        "The value-aware policy deliberately prioritizes margin at risk rather than churn "
        "probability alone. "
        + lift_explanation
        + " Compare average churn probability, average margin, expected value per contact, "
        "observed churn rate, recall, precision, lift, campaign cost, and total expected value. "
        "Higher customer margin can outweigh fewer captured churners under the scenario formula.",
        "",
        "![Policy value decomposition](figures/policy_value_decomposition.png)",
        "",
        "## Sensitivity",
        "",
        f"Across {len(sensitivity)} unique configurations, expected net value ranged from "
        f"{scenario.currency} {sensitivity['expected_net_value'].min():,.2f} to "
        f"{scenario.currency} {sensitivity['expected_net_value'].max():,.2f}. The analysis varies "
        "incremental effect, offer acceptance, and gross-margin scaling.",
        "",
        "## Risks and next steps",
        "",
        "The source contains no treatment assignment or campaign response, so retention "
        "effect and offer acceptance are configurable scenario inputs rather than causal "
        "estimates. A "
        "real next step is a "
        "randomized controlled retention experiment, followed by uplift modeling and segment-level "
        "fairness, consent, deliverability, and capacity checks. A 90-day economic horizon would "
        "require a separately trained 90-day model or survival analysis; the 45-day probability "
        "is not extrapolated.",
    ]
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
