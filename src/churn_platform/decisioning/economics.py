"""Configurable, explicitly non-causal retention economics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from churn_platform.config import dataclass_from_dict, load_yaml


@dataclass(frozen=True)
class EconomicScenario:
    """Business assumptions used to convert risk into scenario value."""

    scenario_name: str = "base_case"
    currency: str = "GBP"
    total_budget: float = 12_000.0
    contact_cost: float = 2.0
    offer_cost_if_accepted: float = 12.0
    offer_acceptance_probability: float = 0.35
    incremental_retention_effect: float = 0.20
    max_contact_fraction: float = 0.15
    economic_horizon_days: int = 45
    gross_margin_rate: float = 0.30
    minimum_customer_margin: float = 20.0
    random_seed: int = 42

    def validate(self) -> None:
        """Reject economic inputs that could produce an unsafe decision contract."""
        if self.total_budget < 0 or self.contact_cost < 0 or self.offer_cost_if_accepted < 0:
            raise ValueError("Budget and costs must be non-negative")
        if not 0 <= self.incremental_retention_effect <= 1:
            raise ValueError("incremental_retention_effect must be in [0, 1]")
        if not 0 <= self.offer_acceptance_probability <= 1:
            raise ValueError("offer_acceptance_probability must be in [0, 1]")
        if not 0 < self.max_contact_fraction <= 1:
            raise ValueError("max_contact_fraction must be in (0, 1]")
        if not 0 <= self.gross_margin_rate <= 1:
            raise ValueError("gross_margin_rate must be in [0, 1]")
        if self.economic_horizon_days <= 0:
            raise ValueError("economic_horizon_days must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return the public scenario contract."""
        return {
            "scenario_name": self.scenario_name,
            "currency": self.currency,
            "total_budget": self.total_budget,
            "contact_cost": self.contact_cost,
            "offer_cost_if_accepted": self.offer_cost_if_accepted,
            "offer_acceptance_probability": self.offer_acceptance_probability,
            "incremental_retention_effect": self.incremental_retention_effect,
            "max_contact_fraction": self.max_contact_fraction,
            "economic_horizon_days": self.economic_horizon_days,
            "gross_margin_rate": self.gross_margin_rate,
            "minimum_customer_margin": self.minimum_customer_margin,
            "random_seed": self.random_seed,
        }


def load_economic_scenario(
    path: str | Path = "configs/decisioning.yaml",
) -> EconomicScenario:
    """Load and validate an economic scenario from YAML."""
    scenario = dataclass_from_dict(EconomicScenario, load_yaml(path))
    scenario.validate()
    return scenario


def estimate_margin_at_risk(
    monetary_value: pd.Series | np.ndarray | float,
    scenario: EconomicScenario,
    feature_history_days: int = 180,
) -> np.ndarray:
    """Estimate gross margin over the configured economic horizon."""
    values = np.asarray(monetary_value, dtype=float)
    annualization = scenario.economic_horizon_days / feature_history_days
    margin = values * scenario.gross_margin_rate * annualization
    return np.maximum(margin, scenario.minimum_customer_margin)


def expected_contact_cost(scenario: EconomicScenario) -> float:
    """Return expected per-contact spend under the explicit acceptance assumption."""
    return scenario.contact_cost + (
        scenario.offer_acceptance_probability * scenario.offer_cost_if_accepted
    )


def validate_horizon_alignment(prediction_horizon_days: int, scenario: EconomicScenario) -> None:
    """Require economic value and predicted risk to refer to the same horizon."""
    if prediction_horizon_days != scenario.economic_horizon_days:
        raise ValueError(
            "Economic and prediction horizons must match; train a separate model or use "
            "survival analysis for a different horizon"
        )


def expected_net_value(
    churn_probability: pd.Series | np.ndarray | float,
    margin_at_risk: pd.Series | np.ndarray | float,
    scenario: EconomicScenario,
) -> np.ndarray:
    """Calculate expected scenario value; this is not an identified causal effect."""
    probability = np.asarray(churn_probability, dtype=float)
    margin = np.asarray(margin_at_risk, dtype=float)
    return probability * scenario.incremental_retention_effect * margin - expected_contact_cost(
        scenario
    )
