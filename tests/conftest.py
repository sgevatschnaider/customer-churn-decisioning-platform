"""Shared deterministic fixtures for unit and integration tests."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from churn_platform.config import load_data_config
from churn_platform.data.fixtures import generate_fixture
from churn_platform.data.ingest import normalize_transactions
from churn_platform.features.snapshots import build_snapshots


@pytest.fixture(scope="session", autouse=True)
def isolated_project_root(tmp_path_factory: pytest.TempPathFactory):
    """Keep fixture-pipeline outputs away from tracked UCI evidence."""
    import churn_platform.config as config_module
    import churn_platform.reproducibility as reproducibility_module

    repository_root = Path(__file__).resolve().parents[1]
    isolated_root = tmp_path_factory.mktemp("project")
    shutil.copytree(repository_root / "configs", isolated_root / "configs")
    shutil.copytree(repository_root / "data" / "fixtures", isolated_root / "data" / "fixtures")
    shutil.copytree(
        repository_root / "src" / "churn_platform",
        isolated_root / "src" / "churn_platform",
    )
    shutil.copy2(repository_root / "requirements.lock", isolated_root / "requirements.lock")
    shutil.copy2(repository_root / "LICENSE", isolated_root / "LICENSE")
    original_config_root = config_module.PROJECT_ROOT
    original_reproducibility_root = reproducibility_module.PROJECT_ROOT
    original_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    tracking_root = Path(tempfile.mkdtemp(prefix="churn-mlflow-"))
    config_module.PROJECT_ROOT = isolated_root
    reproducibility_module.PROJECT_ROOT = isolated_root
    os.environ["MLFLOW_TRACKING_URI"] = tracking_root.as_uri()
    try:
        yield
    finally:
        config_module.PROJECT_ROOT = original_config_root
        reproducibility_module.PROJECT_ROOT = original_reproducibility_root
        if original_tracking_uri is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = original_tracking_uri


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
