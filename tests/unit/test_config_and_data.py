from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from churn_platform.config import ConfigurationError, load_data_config, load_yaml, project_path
from churn_platform.data.download import DownloadError, download_dataset, sha256_file
from churn_platform.data.fixtures import generate_fixture
from churn_platform.data.ingest import ingest_transactions, normalize_transactions
from churn_platform.data.validation import DataValidationError, validate_transactions
from churn_platform.models.train import resolve_tracking_uri


def test_configuration_loads_and_paths_resolve() -> None:
    config = load_data_config()
    assert config.history_days == 180
    assert config.horizon_days == 45
    assert config.eligibility_config().max_recency_days == 81
    assert config.eligibility_config().minimum_invoices == 2
    assert project_path("configs/data.yaml").exists()
    assert load_yaml("configs/model.yaml")["random_seed"] == 42
    with pytest.raises(ConfigurationError):
        load_yaml("configs/does-not-exist.yaml")


def test_fixture_is_deterministic_and_ingestable(tmp_path) -> None:
    first = generate_fixture(tmp_path / "first.csv", customers=12)
    second = generate_fixture(tmp_path / "second.csv", customers=12)
    pd.testing.assert_frame_equal(first, second)
    normalized = ingest_transactions(tmp_path / "first.csv", tmp_path / "normalized.parquet")
    summary = validate_transactions(normalized)
    assert summary.customers == 12
    assert summary.rows == len(normalized)
    assert summary.cancellation_rate > 0
    assert (tmp_path / "normalized.parquet").exists()


def test_normalization_and_validation_failures() -> None:
    with pytest.raises(ValueError, match="Missing transaction columns"):
        normalize_transactions(pd.DataFrame({"InvoiceNo": ["1"]}))
    frame = pd.DataFrame(
        {
            "InvoiceNo": ["1"],
            "StockCode": ["P1"],
            "Description": ["x"],
            "Quantity": [1],
            "InvoiceDate": ["2024-01-01"],
            "UnitPrice": [-1],
            "CustomerID": ["1"],
            "Country": ["France"],
        }
    )
    normalized = normalize_transactions(frame)
    with pytest.raises(DataValidationError, match="negative"):
        validate_transactions(normalized)
    with pytest.raises(ValueError, match="Unsupported"):
        ingest_transactions("LICENSE", "unused.parquet")


def test_archive_checksum_and_integrity(tmp_path) -> None:
    archive = tmp_path / "dataset.zip"
    workbook_bytes = b"synthetic workbook bytes"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("Online Retail.xlsx", workbook_bytes)
    checksum = sha256_file(archive)
    manifest = download_dataset(
        "https://not-called.invalid/dataset.zip",
        archive,
        tmp_path / "Online Retail.xlsx",
        checksum,
    )
    assert manifest["archive_sha256"] == checksum
    assert (tmp_path / "Online Retail.xlsx").read_bytes() == workbook_bytes
    with pytest.raises(DownloadError, match="Checksum mismatch"):
        download_dataset(
            "https://not-called.invalid/dataset.zip",
            archive,
            tmp_path / "copy.xlsx",
            "0" * 64,
        )


def test_mlflow_tracking_uri_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert resolve_tracking_uri("http://explicit:5001") == "http://explicit:5001"
    assert resolve_tracking_uri() == "http://mlflow:5000"
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    assert resolve_tracking_uri().startswith("file:")
