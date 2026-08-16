"""Typed configuration loading with repository-relative path resolution."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
T = TypeVar("T")


class ConfigurationError(ValueError):
    """Raised when a configuration file is missing or invalid."""


def project_path(path: str | Path) -> Path:
    """Return an absolute path, interpreting relative values from the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject ambiguous or missing configuration."""
    config_path = project_path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Expected a YAML mapping in {config_path}")
    return payload


def dataclass_from_dict(cls: type[T], values: dict[str, Any]) -> T:
    """Build a dataclass using only declared fields from a configuration mapping."""
    declared = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in declared})


@dataclass(frozen=True)
class DataConfig:
    """Point-in-time data and temporal split configuration."""

    history_days: int
    horizon_days: int
    cutoffs: dict[str, list[str]]
    fixture_cutoffs: dict[str, list[str]]
    fixture_path: str = "data/fixtures/transactions.csv"
    normalized_path: str = "data/interim/transactions.parquet"
    snapshots_path: str = "data/processed/customer_snapshots.parquet"
    raw_zip_path: str = "data/raw/online_retail.zip"
    raw_xlsx_path: str = "data/raw/Online Retail.xlsx"
    download_url: str = ""

    def split_cutoffs(self, fixture: bool = False) -> dict[str, list[str]]:
        """Return fixture or full-data cutoffs."""
        return self.fixture_cutoffs if fixture else self.cutoffs


def load_data_config(path: str | Path = "configs/data.yaml") -> DataConfig:
    """Load data configuration."""
    return dataclass_from_dict(DataConfig, load_yaml(path))
