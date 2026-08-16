from __future__ import annotations

import pandas as pd
import pytest

from churn_platform.config import load_data_config
from churn_platform.features.build_features import MODEL_FEATURES
from churn_platform.features.snapshots import (
    PointInTimeError,
    assert_point_in_time_integrity,
    build_snapshot,
)


def test_snapshot_features_and_labels_are_point_in_time(
    synthetic_transactions: pd.DataFrame, synthetic_snapshots: pd.DataFrame
) -> None:
    assert set(MODEL_FEATURES).issubset(synthetic_snapshots.columns)
    assert (
        synthetic_snapshots["feature_max_event_date"].le(synthetic_snapshots["cutoff_date"]).all()
    )
    observed = synthetic_snapshots["label_min_event_date"].notna()
    assert (
        synthetic_snapshots.loc[observed, "label_min_event_date"]
        .gt(synthetic_snapshots.loc[observed, "cutoff_date"])
        .all()
    )
    assert synthetic_snapshots["churn"].nunique() == 2
    assert_point_in_time_integrity(synthetic_snapshots, synthetic_transactions, 180)


def test_split_windows_do_not_overlap(synthetic_snapshots: pd.DataFrame) -> None:
    train_end = synthetic_snapshots.loc[
        synthetic_snapshots["split"].eq("train"), "label_window_end"
    ].max()
    validation_cutoff = synthetic_snapshots.loc[
        synthetic_snapshots["split"].eq("validation"), "cutoff_date"
    ].min()
    validation_end = synthetic_snapshots.loc[
        synthetic_snapshots["split"].eq("validation"), "label_window_end"
    ].max()
    test_cutoff = synthetic_snapshots.loc[
        synthetic_snapshots["split"].eq("test"), "cutoff_date"
    ].min()
    assert train_end < validation_cutoff
    assert validation_end < test_cutoff


def test_integrity_detects_lineage_tampering(
    synthetic_transactions: pd.DataFrame, synthetic_snapshots: pd.DataFrame
) -> None:
    tampered = synthetic_snapshots.copy()
    tampered.loc[tampered.index[0], "feature_max_event_date"] = tampered.loc[
        tampered.index[0], "cutoff_date"
    ] + pd.Timedelta(days=1)
    with pytest.raises(PointInTimeError, match="feature uses"):
        assert_point_in_time_integrity(tampered, synthetic_transactions, 180)


def test_incomplete_future_horizon_is_rejected(synthetic_transactions: pd.DataFrame) -> None:
    config = load_data_config()
    with pytest.raises(PointInTimeError, match="incomplete"):
        build_snapshot(
            synthetic_transactions,
            synthetic_transactions["invoice_date"].max() - pd.Timedelta(days=1),
            "test",
            config.history_days,
            config.horizon_days,
        )
