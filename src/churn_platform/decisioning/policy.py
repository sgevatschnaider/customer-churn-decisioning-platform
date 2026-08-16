"""Budget-constrained retention targeting and policy comparison."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import pandas as pd

from churn_platform.decisioning.economics import (
    EconomicScenario,
    estimate_margin_at_risk,
    expected_contact_cost,
    expected_net_value,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PolicyName = Literal["random", "churn_probability", "expected_value"]


@dataclass(frozen=True)
class CampaignCapacity:
    """Financial, operational, and economic limits for one ranked portfolio."""

    budget_based_contact_capacity: int
    operations_based_contact_capacity: int
    economically_eligible_customers: int
    actual_selected_customers: int
    binding_constraint: str
    expected_cost_per_contact: float


def campaign_capacity(
    customers: int,
    scenario: EconomicScenario,
    economically_eligible_customers: int | None = None,
) -> CampaignCapacity:
    """Calculate all campaign limits and identify the constraint that binds."""
    scenario.validate()
    expected_unit_cost = expected_contact_cost(scenario)
    budget_limit = (
        customers
        if expected_unit_cost == 0
        else min(customers, int(np.floor(scenario.total_budget / expected_unit_cost)))
    )
    operations_limit = min(customers, int(np.floor(customers * scenario.max_contact_fraction)))
    economic_limit = (
        customers
        if economically_eligible_customers is None
        else min(customers, max(0, economically_eligible_customers))
    )
    selected = max(0, min(customers, budget_limit, operations_limit, economic_limit))
    binding = []
    if budget_limit == selected:
        binding.append("financial_budget")
    if operations_limit == selected:
        binding.append("operational_capacity")
    if economically_eligible_customers is not None and economic_limit == selected:
        binding.append("positive_expected_value")
    binding_constraint = binding[0] if len(binding) == 1 else "joint:" + "+".join(binding)
    return CampaignCapacity(
        budget_based_contact_capacity=budget_limit,
        operations_based_contact_capacity=operations_limit,
        economically_eligible_customers=economic_limit,
        actual_selected_customers=selected,
        binding_constraint=binding_constraint,
        expected_cost_per_contact=expected_unit_cost,
    )


def contact_capacity(customers: int, scenario: EconomicScenario) -> int:
    """Return the financial/operational capacity for compatibility callers."""
    return campaign_capacity(customers, scenario).actual_selected_customers


def _policy_score(frame: pd.DataFrame, policy: PolicyName, seed: int) -> np.ndarray:
    if policy == "random":
        return np.random.default_rng(seed).random(len(frame))
    if policy == "churn_probability":
        return frame["churn_probability"].to_numpy()
    if policy == "expected_value":
        return frame["expected_net_value"].to_numpy()
    raise ValueError(f"Unsupported policy: {policy}")


def apply_retention_policy(
    customers: pd.DataFrame,
    scenario: EconomicScenario,
    policy: PolicyName = "expected_value",
    feature_history_days: int = 180,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank customers, impose capacity, and explain every recommendation."""
    required = {"customer_id", "churn_probability", "monetary_value"}
    missing = sorted(required - set(customers.columns))
    if missing:
        raise ValueError(f"Missing decisioning columns: {missing}")
    scenario.validate()
    ranked = customers.copy()
    ranked["estimated_margin_at_risk"] = estimate_margin_at_risk(
        ranked["monetary_value"], scenario, feature_history_days
    )
    ranked["value_at_risk"] = ranked["churn_probability"] * ranked["estimated_margin_at_risk"]
    ranked["expected_net_value"] = expected_net_value(
        ranked["churn_probability"], ranked["estimated_margin_at_risk"], scenario
    )
    ranked["economic_eligibility"] = ranked["expected_net_value"].gt(0)
    ranked["policy_score"] = _policy_score(ranked, policy, scenario.random_seed)
    ranked = ranked.sort_values(
        ["policy_score", "customer_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    economic_eligible = ranked["economic_eligibility"]
    capacity = campaign_capacity(
        len(ranked),
        scenario,
        int(economic_eligible.sum()) if policy == "expected_value" else None,
    )
    ranked["policy_rank"] = np.arange(1, len(ranked) + 1)
    ranked["recommended_action"] = "do_not_contact"
    eligible = economic_eligible if policy == "expected_value" else True
    if isinstance(eligible, bool):
        eligible = pd.Series(eligible, index=ranked.index)
    selected_index = ranked.index[eligible][: capacity.actual_selected_customers]
    ranked.loc[selected_index, "recommended_action"] = "contact"
    if policy == "expected_value":
        ranked["selection_reason"] = np.where(
            ranked["recommended_action"].eq("contact"),
            "Highest positive expected net value within portfolio constraints",
            np.where(
                ranked["expected_net_value"].le(0),
                "Non-positive expected net value",
                f"Below the rank allowed by {capacity.binding_constraint}",
            ),
        )
    elif policy == "churn_probability":
        ranked["selection_reason"] = np.where(
            ranked["recommended_action"].eq("contact"),
            "Highest churn probability within portfolio constraints",
            f"Below the rank allowed by {capacity.binding_constraint}",
        )
    else:
        ranked["selection_reason"] = np.where(
            ranked["recommended_action"].eq("contact"),
            "Seeded random benchmark selection",
            "Not selected by seeded random benchmark",
        )
    ranked["economic_scenario"] = scenario.scenario_name

    selected = ranked["recommended_action"].eq("contact")
    observed_churn = int(ranked.loc[selected, "churn"].sum()) if "churn" in ranked.columns else None
    expected_campaign_cost = float(selected.sum()) * capacity.expected_cost_per_contact
    expected_total_value = float(ranked.loc[selected, "expected_net_value"].sum())
    summary: dict[str, Any] = {
        "policy": policy,
        "customers": int(len(ranked)),
        "contact_capacity": capacity.actual_selected_customers,
        "budget_based_contact_capacity": capacity.budget_based_contact_capacity,
        "operations_based_contact_capacity": capacity.operations_based_contact_capacity,
        "economically_eligible_customers": int(economic_eligible.sum()),
        "binding_constraint": capacity.binding_constraint,
        "customers_contacted": int(selected.sum()),
        "actual_selected_customers": int(selected.sum()),
        "contact_rate": float(selected.mean()),
        "expected_net_value": expected_total_value,
        "expected_net_value_per_contact": (
            expected_total_value / int(selected.sum()) if selected.any() else 0.0
        ),
        "estimated_value_at_risk": float(ranked.loc[selected, "value_at_risk"].sum()),
        "average_churn_probability": (
            float(ranked.loc[selected, "churn_probability"].mean()) if selected.any() else 0.0
        ),
        "average_estimated_margin_at_risk": (
            float(ranked.loc[selected, "estimated_margin_at_risk"].mean())
            if selected.any()
            else 0.0
        ),
        "expected_campaign_cost": expected_campaign_cost,
        "remaining_budget": float(scenario.total_budget - expected_campaign_cost),
        "budget_utilization_percentage": (
            expected_campaign_cost / scenario.total_budget * 100 if scenario.total_budget else 0.0
        ),
        "observed_churners_contacted": observed_churn,
        "scenario": scenario.to_dict(),
    }
    if "churn" in ranked.columns:
        total_churners = int(ranked["churn"].sum())
        captured = int(ranked.loc[selected, "churn"].sum())
        precision = float(ranked.loc[selected, "churn"].mean()) if selected.any() else 0.0
        base_rate = float(ranked["churn"].mean())
        summary.update(
            {
                "observed_churners": total_churners,
                "recall_at_budget": captured / total_churners if total_churners else 0.0,
                "precision_at_budget": precision,
                "observed_churn_rate": precision,
                "lift_at_budget": precision / base_rate if base_rate else 0.0,
                "scenario_realized_net_value": float(
                    (
                        ranked.loc[selected, "churn"]
                        * scenario.incremental_retention_effect
                        * ranked.loc[selected, "estimated_margin_at_risk"]
                        - capacity.expected_cost_per_contact
                    ).sum()
                ),
            }
        )
    return ranked, summary


def compare_policies(
    customers: pd.DataFrame,
    scenario: EconomicScenario,
    feature_history_days: int = 180,
    random_repetitions: int = 200,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compare churn and value policies with a repeated seeded random benchmark."""
    summaries = []
    rankings: dict[str, pd.DataFrame] = {}
    for policy in ("random", "churn_probability", "expected_value"):
        ranking, summary = apply_retention_policy(
            customers, scenario, policy=policy, feature_history_days=feature_history_days
        )
        rankings[policy] = ranking
        summary["benchmark_repetitions"] = 1
        if policy == "random" and random_repetitions > 1:
            random_summaries = []
            for offset in range(random_repetitions):
                varied = EconomicScenario(
                    **{**scenario.to_dict(), "random_seed": scenario.random_seed + offset}
                )
                _, random_summary = apply_retention_policy(
                    customers,
                    varied,
                    policy="random",
                    feature_history_days=feature_history_days,
                )
                random_summaries.append(random_summary)
            averaged_keys = (
                "expected_net_value",
                "expected_net_value_per_contact",
                "estimated_value_at_risk",
                "average_churn_probability",
                "average_estimated_margin_at_risk",
                "expected_campaign_cost",
                "remaining_budget",
                "budget_utilization_percentage",
                "observed_churners_contacted",
                "recall_at_budget",
                "precision_at_budget",
                "observed_churn_rate",
                "lift_at_budget",
                "scenario_realized_net_value",
            )
            for key in averaged_keys:
                summary[key] = float(np.mean([item[key] for item in random_summaries]))
            summary["benchmark_repetitions"] = random_repetitions
        summaries.append({key: value for key, value in summary.items() if key != "scenario"})
    return pd.DataFrame(summaries), rankings


def sensitivity_analysis(
    customers: pd.DataFrame,
    scenario: EconomicScenario,
    feature_history_days: int = 180,
) -> pd.DataFrame:
    """Evaluate value-policy robustness across effect, acceptance, and margin assumptions."""
    rows = []
    for incremental_effect in (0.10, scenario.incremental_retention_effect, 0.35):
        for acceptance_probability in (0.20, scenario.offer_acceptance_probability, 0.50):
            for margin_multiplier in (0.75, 1.0, 1.25):
                varied = EconomicScenario(
                    **{
                        **scenario.to_dict(),
                        "scenario_name": "sensitivity",
                        "incremental_retention_effect": incremental_effect,
                        "offer_acceptance_probability": acceptance_probability,
                        "gross_margin_rate": min(
                            1.0, scenario.gross_margin_rate * margin_multiplier
                        ),
                    }
                )
                _, summary = apply_retention_policy(
                    customers,
                    varied,
                    policy="expected_value",
                    feature_history_days=feature_history_days,
                )
                rows.append(
                    {
                        "incremental_retention_effect": incremental_effect,
                        "offer_acceptance_probability": acceptance_probability,
                        "margin_multiplier": margin_multiplier,
                        "customers_contacted": summary["customers_contacted"],
                        "expected_net_value": summary["expected_net_value"],
                        "expected_campaign_cost": summary["expected_campaign_cost"],
                        "binding_constraint": summary["binding_constraint"],
                    }
                )
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def anonymize_customer_id(identifier: object) -> str:
    """Create a stable non-reversible identifier for published decision artifacts."""
    return hashlib.sha256(f"churn-platform:{identifier}".encode()).hexdigest()[:16]


def save_decision_outputs(
    rankings: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    artifact_path: str | Path,
    figures_directory: str | Path,
) -> None:
    """Persist the value-aware ranking without publishing source customer identifiers."""
    value_ranking = rankings["expected_value"].copy()
    value_ranking["customer_id"] = value_ranking["customer_id"].map(anonymize_customer_id)
    public_columns = [
        "customer_id",
        "churn_probability",
        "value_at_risk",
        "expected_net_value",
        "economic_eligibility",
        "recommended_action",
        "selection_reason",
        "policy_rank",
        "economic_scenario",
    ]
    destination = Path(artifact_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    value_ranking[public_columns].to_csv(destination, index=False)
    comparison.to_csv(destination.parent / "policy_comparison.csv", index=False)
    sensitivity.to_csv(destination.parent / "sensitivity_analysis.csv", index=False)

    figures = Path(figures_directory)
    figures.mkdir(parents=True, exist_ok=True)
    plot_columns = ["expected_net_value"]
    if "scenario_realized_net_value" in comparison.columns:
        plot_columns.append("scenario_realized_net_value")
    comparison.set_index("policy")[plot_columns].plot(kind="bar", figsize=(7, 4))
    plt.ylabel("Value (GBP)")
    plt.title("Retention policy comparison")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures / "policy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    decomposition = comparison.set_index("policy")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    decomposition["average_churn_probability"].plot(kind="bar", ax=axes[0], color="#457B9D")
    axes[0].set_title("Average churn probability")
    axes[0].set_ylabel("Probability")
    decomposition["average_estimated_margin_at_risk"].plot(kind="bar", ax=axes[1], color="#E9C46A")
    axes[1].set_title("Average margin at risk")
    axes[1].set_ylabel("GBP")
    decomposition["expected_net_value_per_contact"].plot(kind="bar", ax=axes[2], color="#2A9D8F")
    axes[2].set_title("Expected net value per contact")
    axes[2].set_ylabel("GBP")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    figure.suptitle("Why value-aware targeting can trade churn lift for margin")
    figure.tight_layout()
    figure.savefig(figures / "policy_value_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    budgets = np.linspace(0.05, 0.50, 10)
    values = []
    source = rankings["expected_value"].copy()
    for fraction in budgets:
        count = max(1, int(len(source) * fraction))
        values.append(
            float(
                source.nlargest(count, "expected_net_value")["expected_net_value"]
                .clip(lower=0)
                .sum()
            )
        )
    plt.figure(figsize=(6, 4))
    plt.plot(budgets, values, marker="o", color="#2A9D8F")
    plt.xlabel("Maximum contact fraction")
    plt.ylabel("Expected net value (GBP)")
    plt.title("Scenario value by contact budget")
    plt.tight_layout()
    plt.savefig(figures / "value_by_budget.png", dpi=150, bbox_inches="tight")
    plt.close()
