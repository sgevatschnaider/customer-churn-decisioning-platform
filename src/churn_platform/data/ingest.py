"""Normalize raw UCI and fixture transactions into a stable internal schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn_platform.config import project_path

CANONICAL_COLUMNS = [
    "invoice_no",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "unit_price",
    "customer_id",
    "country",
]

UCI_COLUMN_MAP = {
    "InvoiceNo": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "UnitPrice": "unit_price",
    "CustomerID": "customer_id",
    "Country": "country",
}


def normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize names, identifiers, dates, and numeric fields."""
    normalized = frame.rename(columns=UCI_COLUMN_MAP).copy()
    missing = sorted(set(CANONICAL_COLUMNS) - set(normalized.columns))
    if missing:
        raise ValueError(f"Missing transaction columns: {missing}")
    normalized = normalized[CANONICAL_COLUMNS]
    normalized["invoice_date"] = pd.to_datetime(normalized["invoice_date"], errors="coerce")
    for column in ("quantity", "unit_price"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["customer_id"] = (
        normalized["customer_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    )
    for column in ("invoice_no", "stock_code", "description", "country"):
        normalized[column] = normalized[column].astype("string")
    normalized["is_cancellation"] = normalized["invoice_no"].str.upper().str.startswith(
        "C", na=False
    ) | normalized["quantity"].lt(0)
    normalized["line_value"] = normalized["quantity"] * normalized["unit_price"]
    normalized["event_id"] = pd.util.hash_pandas_object(
        normalized[["invoice_no", "stock_code", "invoice_date", "customer_id"]], index=False
    ).astype("string")
    return normalized.sort_values("invoice_date").reset_index(drop=True)


def ingest_transactions(source_path: str | Path, destination_path: str | Path) -> pd.DataFrame:
    """Read XLSX/CSV/Parquet transactions, normalize them, and persist Parquet."""
    source = project_path(source_path)
    destination = project_path(destination_path)
    if not source.exists():
        raise FileNotFoundError(f"Transaction source does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".xlsx":
        raw = pd.read_excel(source, engine="openpyxl")
    elif suffix == ".csv":
        raw = pd.read_csv(source)
    elif suffix == ".parquet":
        raw = pd.read_parquet(source)
    else:
        raise ValueError(f"Unsupported transaction format: {suffix}")
    normalized = normalize_transactions(raw)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(destination, index=False)
    return normalized
