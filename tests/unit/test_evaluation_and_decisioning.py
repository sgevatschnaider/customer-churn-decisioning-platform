from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_platform.decisioning.economics import (
    EconomicScenario,
    estimate_margin_at_risk,
    expected_contact_cost,
    expected_net_value,
    validate_horizon_alignment,
)
from churn_platform.decisioning.policy import (
    anonymize_customer_id,
    apply_retention_policy,
    campaign_capacity,
    compare_policies,
    contact_capacity,
    save_decision_outputs,
    sensitivity_analysis,
)
from churn_platform.models.evaluate import budget_selection, evaluate_predictions


def customer_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(20)],
            "churn_probability": np.linspace(0.05, 0.95, 20),
            "monetary_value": np.linspace(100, 5_000, 20),
            "churn": [0] * 10 + [1] * 10,
        }
    )


def test_budget_metrics_are_exact() -> None:
    target = np.array([0, 0, 1, 1, 1])
    probability = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    selection = budget_selection(probability, 0.4)
    metrics = evaluate_predictions(target, probability, 0.4)
    assert selection.sum() == 2
    assert metrics["precision_at_budget"] == 1.0
    assert metrics["recall_at_budget"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"] == [[2, 0], [1, 2]]
    with pytest.raises(ValueError):
        budget_selection(probability, 0)


def test_economic_formula_and_capacity() -> None:
    scenario = EconomicScenario(total_budget=28, max_contact_fraction=0.5)
    margin = estimate_margin_at_risk([100, 1_000], scenario)
    values = expected_net_value([0.1, 0.9], margin, scenario)
    assert margin.tolist() == [20.0, 75.0]
    assert values[1] > values[0]
    assert expected_contact_cost(scenario) == pytest.approx(6.2)
    assert contact_capacity(20, scenario) == 4
    expected = 0.9 * scenario.incremental_retention_effect * 75 - 6.2
    assert values[1] == pytest.approx(expected)
    with pytest.raises(ValueError):
        EconomicScenario(incremental_retention_effect=1.1).validate()
    validate_horizon_alignment(45, scenario)
    with pytest.raises(ValueError, match="horizons"):
        validate_horizon_alignment(90, scenario)


@pytest.mark.parametrize(
    ("scenario", "economic_count", "expected_binding"),
    [
        (
            EconomicScenario(total_budget=10, max_contact_fraction=1.0),
            100,
            "financial_budget",
        ),
        (
            EconomicScenario(total_budget=100_000, max_contact_fraction=0.10),
            100,
            "operational_capacity",
        ),
        (
            EconomicScenario(total_budget=100_000, max_contact_fraction=1.0),
            3,
            "positive_expected_value",
        ),
    ],
)
def test_campaign_binding_constraints(
    scenario: EconomicScenario, economic_count: int, expected_binding: str
) -> None:
    capacity = campaign_capacity(100, scenario, economic_count)
    assert capacity.binding_constraint == expected_binding
    assert capacity.actual_selected_customers == min(
        capacity.budget_based_contact_capacity,
        capacity.operations_based_contact_capacity,
        capacity.economically_eligible_customers,
    )


def test_policy_comparison_sensitivity_and_public_output(tmp_path) -> None:
    scenario = EconomicScenario(total_budget=10_000, max_contact_fraction=0.2)
    scores = customer_scores()
    comparison, rankings = compare_policies(scores, scenario)
    assert set(comparison["policy"]) == {"random", "churn_probability", "expected_value"}
    ranking, summary = apply_retention_policy(scores, scenario)
    assert summary["customers_contacted"] <= 4
    assert summary["expected_campaign_cost"] <= scenario.total_budget
    assert summary["remaining_budget"] >= 0
    assert summary["binding_constraint"]
    assert (
        ranking.loc[ranking["recommended_action"].eq("contact"), "expected_net_value"].gt(0).all()
    )
    sensitivity = sensitivity_analysis(scores, scenario)
    assert len(sensitivity) > 3
    destination = tmp_path / "retention_targets.csv"
    save_decision_outputs(rankings, comparison, sensitivity, destination, tmp_path / "figures")
    public = pd.read_csv(destination, dtype={"customer_id": str})
    assert "C19" not in set(public["customer_id"])
    assert len(anonymize_customer_id("C19")) == 16
    assert (tmp_path / "figures" / "policy_comparison.png").exists()
