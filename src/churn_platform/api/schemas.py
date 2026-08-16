"""Validated API request and response contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Point-in-time customer features accepted by the model."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, max_length=128)
    recency_days: float = Field(ge=0, le=10_000)
    purchase_frequency: float = Field(ge=0, le=10_000)
    monetary_value: float = Field(ge=0, le=1_000_000_000)
    average_order_value: float = Field(ge=0, le=100_000_000)
    customer_tenure_days: float = Field(ge=0, le=100_000)
    number_of_invoices: float = Field(ge=0, le=10_000_000)
    number_of_unique_products: float = Field(ge=0, le=10_000_000)
    purchase_regularity_days: float | None = Field(default=None, ge=0, le=100_000)
    recent_purchasing_trend: float = Field(ge=-1_000_000, le=1_000_000)
    cancellation_rate: float = Field(ge=0, le=1)
    quantity_purchased: float = Field(ge=0, le=1_000_000_000)
    geographic_segment: str = Field(min_length=1, max_length=64)
    recent_spend: float = Field(ge=0, le=1_000_000_000)
    historical_spend: float = Field(ge=0, le=1_000_000_000)
    spend_change_ratio: float = Field(ge=0, le=1_000_000)
    recent_invoice_count: float = Field(ge=0, le=10_000_000)
    historical_invoice_count: float = Field(ge=0, le=10_000_000)
    frequency_change_ratio: float = Field(ge=0, le=1_000_000)


class PredictResponse(BaseModel):
    """Calibrated churn score with model and validation metadata."""

    customer_id: str
    churn_probability: float = Field(ge=0, le=1)
    model_version: str
    scoring_timestamp: str
    warnings: list[str]


class DecisionResponse(PredictResponse):
    """Customer score augmented with scenario-based economic decisioning."""

    estimated_value_at_risk: float
    expected_net_value: float
    recommended_action: Literal["contact", "do_not_contact"]
    reason: str
    economic_scenario: dict[str, Any]
