"""Leakage-safe customer snapshot and future churn label construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise

import pandas as pd

from churn_platform.features.build_features import aggregate_customer_features

SPLIT_ORDER = ("train", "validation", "test")


class PointInTimeError(ValueError):
    """Raised when temporal lineage or split isolation is violated."""


def _positive_purchases(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        ~frame["is_cancellation"] & frame["quantity"].gt(0) & frame["unit_price"].ge(0)
    ]


def build_snapshot(
    transactions: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    split: str,
    history_days: int,
    horizon_days: int,
) -> pd.DataFrame:
    """Build one point-in-time customer observation table and future-only label."""
    cutoff_timestamp = pd.Timestamp(cutoff)
    feature_start = cutoff_timestamp - pd.Timedelta(days=history_days)
    label_start = cutoff_timestamp + pd.Timedelta(nanoseconds=1)
    label_end = cutoff_timestamp + pd.Timedelta(days=horizon_days)
    maximum_observed = transactions["invoice_date"].max()
    if label_end > maximum_observed:
        raise PointInTimeError(
            f"Cutoff {cutoff_timestamp.date()} has an incomplete {horizon_days}-day horizon; "
            f"data ends {maximum_observed.date()}"
        )

    history = transactions.loc[
        transactions["invoice_date"].gt(feature_start)
        & transactions["invoice_date"].le(cutoff_timestamp)
        & transactions["customer_id"].notna()
    ].copy()
    future = transactions.loc[
        transactions["invoice_date"].gt(cutoff_timestamp)
        & transactions["invoice_date"].le(label_end)
        & transactions["customer_id"].notna()
    ].copy()
    features = aggregate_customer_features(history, cutoff_timestamp)
    if features.empty:
        raise PointInTimeError(f"No eligible customers at cutoff {cutoff_timestamp.date()}")

    future_positive = _positive_purchases(future)
    future_lineage = (
        future_positive.groupby("customer_id", observed=True)
        .agg(
            label_min_event_date=("invoice_date", "min"),
            label_max_event_date=("invoice_date", "max"),
            label_event_count=("event_id", "nunique"),
        )
        .reset_index()
    )
    features = features.merge(future_lineage, on="customer_id", how="left")
    features["churn"] = features["label_event_count"].isna().astype(int)
    features["label_event_count"] = features["label_event_count"].fillna(0).astype(int)
    features["cutoff_date"] = cutoff_timestamp
    features["feature_window_start"] = feature_start
    features["label_window_start"] = label_start
    features["label_window_end"] = label_end
    features["split"] = split
    return features


def build_snapshots(
    transactions: pd.DataFrame,
    split_cutoffs: Mapping[str, Sequence[str]],
    history_days: int,
    horizon_days: int,
) -> pd.DataFrame:
    """Build all configured temporal splits in deterministic order."""
    snapshots = []
    for split in SPLIT_ORDER:
        for cutoff in split_cutoffs.get(split, []):
            snapshots.append(
                build_snapshot(transactions, cutoff, split, history_days, horizon_days)
            )
    if not snapshots:
        raise PointInTimeError("No snapshot cutoffs were configured")
    result = pd.concat(snapshots, ignore_index=True)
    assert_point_in_time_integrity(result, transactions, history_days)
    return result.sort_values(["cutoff_date", "customer_id"]).reset_index(drop=True)


def assert_point_in_time_integrity(
    snapshots: pd.DataFrame,
    transactions: pd.DataFrame,
    history_days: int,
) -> None:
    """Assert feature/label boundaries, event disjointness, and split isolation."""
    required = {
        "cutoff_date",
        "feature_max_event_date",
        "label_min_event_date",
        "label_window_start",
        "label_window_end",
        "split",
    }
    missing = sorted(required - set(snapshots.columns))
    if missing:
        raise PointInTimeError(f"Snapshot lineage columns are missing: {missing}")
    if snapshots["feature_max_event_date"].gt(snapshots["cutoff_date"]).any():
        raise PointInTimeError("At least one feature uses an event after its cutoff")
    observed_labels = snapshots["label_min_event_date"].notna()
    if (
        snapshots.loc[observed_labels, "label_min_event_date"]
        .le(snapshots.loc[observed_labels, "cutoff_date"])
        .any()
    ):
        raise PointInTimeError("At least one label uses an event at or before its cutoff")
    if (
        snapshots.loc[observed_labels, "label_min_event_date"]
        .lt(snapshots.loc[observed_labels, "label_window_start"])
        .any()
    ):
        raise PointInTimeError("Label lineage precedes the configured label window")

    for cutoff, rows in snapshots.groupby("cutoff_date", observed=True):
        label_end = rows["label_window_end"].iloc[0]
        feature_start = cutoff - pd.Timedelta(days=history_days)
        feature_events = set(
            transactions.loc[
                transactions["invoice_date"].gt(feature_start)
                & transactions["invoice_date"].le(cutoff),
                "event_id",
            ]
        )
        label_events = set(
            transactions.loc[
                transactions["invoice_date"].gt(cutoff)
                & transactions["invoice_date"].le(label_end),
                "event_id",
            ]
        )
        if feature_events & label_events:
            raise PointInTimeError(f"Feature and label events overlap at cutoff {cutoff}")

    split_ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for split in SPLIT_ORDER:
        split_rows = snapshots.loc[snapshots["split"].eq(split)]
        if not split_rows.empty:
            split_ranges[split] = (
                split_rows["cutoff_date"].min(),
                split_rows["label_window_end"].max(),
            )
    for earlier, later in pairwise(SPLIT_ORDER):
        if earlier in split_ranges and later in split_ranges:
            earlier_end = split_ranges[earlier][1]
            later_cutoff = split_ranges[later][0]
            if earlier_end >= later_cutoff:
                raise PointInTimeError(
                    f"{earlier} labels end {earlier_end.date()}, not before the "
                    f"{later} cutoff {later_cutoff.date()}"
                )
