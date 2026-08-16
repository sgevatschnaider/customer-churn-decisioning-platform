"""Active-customer eligibility derived without validation or test outcomes."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def derive_recency_threshold(
    transactions: pd.DataFrame,
    training_cutoff: str | pd.Timestamp,
    quantile: float = 0.90,
) -> dict[str, Any]:
    """Derive an integer recency ceiling from training-period repeat-purchase gaps.

    The calculation uses positive purchases at or before the final training cutoff only.
    Validation and test behavior are never consulted.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    cutoff = pd.Timestamp(training_cutoff)
    positive = transactions.loc[
        transactions["invoice_date"].le(cutoff)
        & transactions["customer_id"].notna()
        & ~transactions["is_cancellation"]
        & transactions["quantity"].gt(0)
        & transactions["unit_price"].ge(0)
    ].drop_duplicates("event_id", keep="first")
    invoices = (
        positive.groupby(["customer_id", "invoice_no"], observed=True)["invoice_date"]
        .min()
        .reset_index()
        .sort_values(["customer_id", "invoice_date", "invoice_no"])
    )
    gaps = (
        invoices.groupby("customer_id", observed=True)["invoice_date"]
        .diff()
        .dt.total_seconds()
        .div(86_400)
        .dropna()
    )
    if gaps.empty:
        raise ValueError("Training period contains no repeat-purchase intervals")
    observed_quantile = float(gaps.quantile(quantile))
    threshold = int(math.ceil(observed_quantile))
    invoice_counts = invoices.groupby("customer_id", observed=True)["invoice_no"].nunique()
    return {
        "training_cutoff": cutoff.date().isoformat(),
        "quantile": quantile,
        "observed_interval_days": observed_quantile,
        "max_recency_days": threshold,
        "repeat_intervals": int(len(gaps)),
        "training_customers": int(invoices["customer_id"].nunique()),
        "repeat_customers": int(invoice_counts.ge(2).sum()),
    }


def apply_customer_eligibility(
    candidates: pd.DataFrame,
    *,
    max_recency_days: int,
    minimum_invoices: int,
    observation_history_complete: bool,
    future_label_horizon_complete: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter a snapshot to active repeat buyers and report every exclusion criterion."""
    required = {"recency_days", "number_of_invoices", "churn"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Missing eligibility columns: {missing}")
    evaluated = candidates.copy()
    failures = {
        "incomplete_observation_history": pd.Series(
            not observation_history_complete, index=evaluated.index
        ),
        "incomplete_future_label_horizon": pd.Series(
            not future_label_horizon_complete, index=evaluated.index
        ),
        "recency_above_maximum": evaluated["recency_days"].gt(max_recency_days),
        "insufficient_invoices": evaluated["number_of_invoices"].lt(minimum_invoices),
    }
    failure_frame = pd.DataFrame(failures, index=evaluated.index)
    evaluated["is_eligible"] = ~failure_frame.any(axis=1)
    evaluated["eligibility_reason"] = failure_frame.apply(
        lambda row: "eligible" if not row.any() else ";".join(row.index[row.to_numpy()].tolist()),
        axis=1,
    )

    primary_reason = pd.Series("eligible", index=evaluated.index, dtype="string")
    unassigned = ~evaluated["is_eligible"]
    for reason in failures:
        matches = unassigned & failure_frame[reason]
        primary_reason.loc[matches] = reason
        unassigned &= ~matches

    eligible = evaluated.loc[evaluated["is_eligible"]].copy()
    summary = {
        "total_customers_before_eligibility": int(len(evaluated)),
        "eligible_customers": int(len(eligible)),
        "excluded_customers": int((~evaluated["is_eligible"]).sum()),
        "eligible_churn_prevalence": float(eligible["churn"].mean()) if len(eligible) else None,
        "criteria": {
            "max_recency_days": max_recency_days,
            "minimum_invoices": minimum_invoices,
            "observation_history_complete": observation_history_complete,
            "future_label_horizon_complete": future_label_horizon_complete,
        },
        "criterion_failure_counts": {reason: int(mask.sum()) for reason, mask in failures.items()},
        "primary_exclusion_reasons": {
            str(reason): int(count)
            for reason, count in primary_reason.loc[~evaluated["is_eligible"]]
            .value_counts()
            .items()
        },
    }
    return eligible, summary


def assert_eligibility_integrity(
    snapshots: pd.DataFrame, max_recency_days: int, minimum_invoices: int
) -> None:
    """Fail closed when an ineligible observation reaches modeling or targeting."""
    required = {"is_eligible", "recency_days", "number_of_invoices"}
    missing = sorted(required - set(snapshots.columns))
    if missing:
        raise ValueError(f"Eligibility lineage columns are missing: {missing}")
    if not snapshots["is_eligible"].all():
        raise ValueError("Ineligible customers are present in the snapshot output")
    if snapshots["recency_days"].gt(max_recency_days).any():
        raise ValueError("A snapshot exceeds the configured recency ceiling")
    if snapshots["number_of_invoices"].lt(minimum_invoices).any():
        raise ValueError("A snapshot has fewer invoices than the eligibility minimum")
