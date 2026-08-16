"""Business feature definitions for customer-level churn snapshots."""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "recency_days",
    "purchase_frequency",
    "monetary_value",
    "average_order_value",
    "customer_tenure_days",
    "number_of_invoices",
    "number_of_unique_products",
    "purchase_regularity_days",
    "recent_purchasing_trend",
    "cancellation_rate",
    "quantity_purchased",
    "recent_spend",
    "historical_spend",
    "spend_change_ratio",
    "recent_invoice_count",
    "historical_invoice_count",
    "frequency_change_ratio",
]
CATEGORICAL_FEATURES = ["geographic_segment"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def geographic_segment(country: pd.Series) -> pd.Series:
    """Map country to a stable low-cardinality commercial segment."""
    europe = {
        "Austria",
        "Belgium",
        "Denmark",
        "Finland",
        "France",
        "Germany",
        "Iceland",
        "Ireland",
        "Italy",
        "Netherlands",
        "Norway",
        "Portugal",
        "Spain",
        "Sweden",
        "Switzerland",
    }
    return pd.Series(
        np.select(
            [country.eq("United Kingdom"), country.isin(europe)],
            ["United Kingdom", "Europe"],
            default="Rest of world",
        ),
        index=country.index,
        dtype="string",
    )


def _invoice_regularity(invoice_dates: pd.Series) -> float:
    unique_dates = pd.Series(pd.to_datetime(invoice_dates).drop_duplicates()).sort_values()
    if len(unique_dates) < 3:
        return float("nan")
    return float(unique_dates.diff().dropna().dt.total_seconds().div(86_400).std(ddof=0))


def aggregate_customer_features(
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
    recent_window_days: int = 45,
) -> pd.DataFrame:
    """Aggregate business features using events at or before the cutoff only."""
    history = history.drop_duplicates("event_id", keep="first")
    positive = history.loc[
        ~history["is_cancellation"] & history["quantity"].gt(0) & history["unit_price"].ge(0)
    ].copy()
    if positive.empty:
        return pd.DataFrame(columns=["customer_id", *MODEL_FEATURES])
    positive["positive_value"] = positive["quantity"] * positive["unit_price"]

    grouped = positive.groupby("customer_id", observed=True)
    customer = grouped.agg(
        monetary_value=("positive_value", "sum"),
        quantity_purchased=("quantity", "sum"),
        number_of_invoices=("invoice_no", "nunique"),
        number_of_unique_products=("stock_code", "nunique"),
        first_purchase_date=("invoice_date", "min"),
        last_purchase_date=("invoice_date", "max"),
        feature_max_event_date=("invoice_date", "max"),
        feature_event_count=("event_id", "nunique"),
    ).reset_index()

    invoice_values = (
        positive.groupby(["customer_id", "invoice_no"], observed=True)
        .agg(invoice_value=("positive_value", "sum"), invoice_date=("invoice_date", "min"))
        .reset_index()
    )
    invoice_stats = (
        invoice_values.groupby("customer_id", observed=True)
        .agg(average_order_value=("invoice_value", "mean"))
        .reset_index()
    )
    regularity = (
        invoice_values.groupby("customer_id", observed=True)["invoice_date"]
        .apply(_invoice_regularity)
        .rename("purchase_regularity_days")
        .reset_index()
    )

    cancellations = (
        history.groupby("customer_id", observed=True)["is_cancellation"]
        .mean()
        .rename("cancellation_rate")
        .reset_index()
    )
    latest_country = (
        history.sort_values("invoice_date")
        .drop_duplicates("customer_id", keep="last")[["customer_id", "country"]]
        .copy()
    )
    latest_country["geographic_segment"] = geographic_segment(latest_country["country"])

    recent_start = cutoff - pd.Timedelta(days=recent_window_days)
    historical_start = cutoff - pd.Timedelta(days=2 * recent_window_days)
    recent = positive.loc[positive["invoice_date"].gt(recent_start)]
    previous = positive.loc[
        positive["invoice_date"].gt(historical_start) & positive["invoice_date"].le(recent_start)
    ]

    def window_aggregation(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        values = (
            frame.groupby("customer_id", observed=True)
            .agg(
                **{
                    f"{prefix}_spend": ("positive_value", "sum"),
                    f"{prefix}_invoice_count": ("invoice_no", "nunique"),
                }
            )
            .reset_index()
        )
        return values

    customer = customer.merge(invoice_stats, on="customer_id", how="left")
    customer = customer.merge(regularity, on="customer_id", how="left")
    customer = customer.merge(cancellations, on="customer_id", how="left")
    customer = customer.merge(
        latest_country[["customer_id", "geographic_segment"]], on="customer_id", how="left"
    )
    customer = customer.merge(window_aggregation(recent, "recent"), on="customer_id", how="left")
    customer = customer.merge(
        window_aggregation(previous, "historical"), on="customer_id", how="left"
    )

    fill_zero = [
        "recent_spend",
        "historical_spend",
        "recent_invoice_count",
        "historical_invoice_count",
    ]
    customer[fill_zero] = customer[fill_zero].fillna(0.0)
    customer["recency_days"] = (cutoff - customer["last_purchase_date"]).dt.total_seconds() / 86_400
    customer["customer_tenure_days"] = (
        cutoff - customer["first_purchase_date"]
    ).dt.total_seconds() / 86_400
    active_months = np.maximum(customer["customer_tenure_days"] / 30.0, 1.0)
    customer["purchase_frequency"] = customer["number_of_invoices"] / active_months
    customer["recent_purchasing_trend"] = (
        customer["recent_spend"] - customer["historical_spend"]
    ) / (customer["historical_spend"].abs() + 1.0)
    customer["spend_change_ratio"] = (customer["recent_spend"] + 1.0) / (
        customer["historical_spend"] + 1.0
    )
    customer["frequency_change_ratio"] = (customer["recent_invoice_count"] + 1.0) / (
        customer["historical_invoice_count"] + 1.0
    )
    return customer
