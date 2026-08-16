"""Shared deterministic fixtures for unit and integration tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn_platform.config import load_data_config
from churn_platform.data.fixtures import generate_fixture
from churn_platform.data.ingest import normalize_transactions
from churn_platform.features.snapshots import build_snapshots


@pytest.fixture(scope="session")
def synthetic_transactions(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    path = tmp_path_factory.mktemp("fixture") / "transactions.csv"
    return normalize_transactions(generate_fixture(path, customers=120))


@pytest.fixture(scope="session")
def synthetic_snapshots(synthetic_transactions: pd.DataFrame) -> pd.DataFrame:
    config = load_data_config()
    return build_snapshots(
        synthetic_transactions,
        config.fixture_cutoffs,
        config.history_days,
        config.horizon_days,
    )


@pytest.fixture(scope="session")
def pipeline_result() -> dict[str, object]:
    from churn_platform.pipeline import run_pipeline

    return run_pipeline("fixture")


@pytest.fixture(scope="session")
def project_root() -> Path:
    from churn_platform.config import PROJECT_ROOT

    return PROJECT_ROOT
