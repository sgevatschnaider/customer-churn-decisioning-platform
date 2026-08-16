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
    offer_cost: float = 12.0
    retention_probability: float = 0.25
    max_contact_fraction: float = 0.15
    economic_horizon_days: int = 90
    gross_margin_rate: float = 0.30
    minimum_customer_margin: float = 20.0
    random_seed: int = 42

    def validate(self) -> None:
        """Reject economic inputs that could produce an unsafe decision contract."""
        if self.total_budget < 0 or self.contact_cost < 0 or self.offer_cost < 0:
            raise ValueError("Budget and costs must be non-negative")
        if not 0 <= self.retention_probability <= 1:
            raise ValueError("retention_probability must be in [0, 1]")
        if not 0 < self.max_contact_fraction <= 1:
            raise ValueError("max_contact_fraction must be in (0, 1]")
        if not 0 <= self.gross_margin_rate <= 1:
            raise ValueError("gross_margin_rate must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Return the public scenario contract."""
        return {
            "scenario_name": self.scenario_name,
            "currency": self.currency,
            "total_budget": self.total_budget,
            "contact_cost": self.contact_cost,
            "offer_cost": self.offer_cost,
            "retention_probability": self.retention_probability,
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


def expected_net_value(
    churn_probability: pd.Series | np.ndarray | float,
    margin_at_risk: pd.Series | np.ndarray | float,
    scenario: EconomicScenario,
) -> np.ndarray:
    """Calculate expected scenario value; this is not an identified causal effect."""
    probability = np.asarray(churn_probability, dtype=float)
    margin = np.asarray(margin_at_risk, dtype=float)
    expected_offer_cost = scenario.retention_probability * scenario.offer_cost
    return (
        probability * scenario.retention_probability * margin
        - scenario.contact_cost
        - expected_offer_cost
    )
