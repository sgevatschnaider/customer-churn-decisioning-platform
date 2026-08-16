"""Budget-constrained retention targeting and policy comparison."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import pandas as pd

from churn_platform.decisioning.economics import (
    EconomicScenario,
    estimate_margin_at_risk,
    expected_net_value,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PolicyName = Literal["random", "churn_probability", "expected_value"]


def contact_capacity(customers: int, scenario: EconomicScenario) -> int:
    """Return capacity constrained by both customer fraction and total budget."""
    fraction_limit = int(np.floor(customers * scenario.max_contact_fraction))
    worst_case_unit_cost = scenario.contact_cost + scenario.offer_cost
    budget_limit = (
        customers
        if worst_case_unit_cost == 0
        else int(np.floor(scenario.total_budget / worst_case_unit_cost))
    )
    return max(0, min(fraction_limit, budget_limit, customers))


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
    ranked["policy_score"] = _policy_score(ranked, policy, scenario.random_seed)
    ranked = ranked.sort_values(
        ["policy_score", "customer_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    capacity = contact_capacity(len(ranked), scenario)
    ranked["policy_rank"] = np.arange(1, len(ranked) + 1)
    ranked["recommended_action"] = "do_not_contact"
    eligible = ranked["expected_net_value"].gt(0) if policy == "expected_value" else True
    if isinstance(eligible, bool):
        eligible = pd.Series(eligible, index=ranked.index)
    selected_index = ranked.index[eligible][:capacity]
    ranked.loc[selected_index, "recommended_action"] = "contact"
    if policy == "expected_value":
        ranked["selection_reason"] = np.where(
            ranked["recommended_action"].eq("contact"),
            "Highest positive expected net value within budget",
            np.where(
                ranked["expected_net_value"].le(0),
                "Non-positive expected net value",
                "Below the budget-constrained value rank",
            ),
        )
    elif policy == "churn_probability":
        ranked["selection_reason"] = np.where(
            ranked["recommended_action"].eq("contact"),
            "Highest churn probability within budget",
            "Below the budget-constrained churn rank",
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
    summary: dict[str, Any] = {
        "policy": policy,
        "customers": int(len(ranked)),
        "contact_capacity": capacity,
        "customers_contacted": int(selected.sum()),
        "contact_rate": float(selected.mean()),
        "expected_net_value": float(ranked.loc[selected, "expected_net_value"].sum()),
        "estimated_value_at_risk": float(ranked.loc[selected, "value_at_risk"].sum()),
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
                "lift_at_budget": precision / base_rate if base_rate else 0.0,
                "scenario_realized_net_value": float(
                    (
                        ranked.loc[selected, "churn"]
                        * scenario.retention_probability
                        * ranked.loc[selected, "estimated_margin_at_risk"]
                        - scenario.contact_cost
                        - scenario.retention_probability * scenario.offer_cost
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
                "estimated_value_at_risk",
                "observed_churners_contacted",
                "recall_at_budget",
                "precision_at_budget",
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
    """Evaluate value-policy robustness across retention, offer, and margin assumptions."""
    rows = []
    for retention_probability in (0.10, scenario.retention_probability, 0.40):
        for offer_cost in (5.0, scenario.offer_cost, 25.0):
            for margin_multiplier in (0.75, 1.0, 1.25):
                varied = EconomicScenario(
                    **{
                        **scenario.to_dict(),
                        "scenario_name": "sensitivity",
                        "retention_probability": retention_probability,
                        "offer_cost": offer_cost,
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
                        "retention_probability": retention_probability,
                        "offer_cost": offer_cost,
                        "margin_multiplier": margin_multiplier,
                        "customers_contacted": summary["customers_contacted"],
                        "expected_net_value": summary["expected_net_value"],
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
