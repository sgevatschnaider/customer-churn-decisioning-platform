"""Strict raw transaction validation and quality summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

REQUIRED_COLUMNS = {
    "invoice_no",
    "stock_code",
    "quantity",
    "invoice_date",
    "unit_price",
    "customer_id",
    "country",
    "is_cancellation",
    "line_value",
    "event_id",
}


class DataValidationError(ValueError):
    """Raised when a dataset violates a blocking quality rule."""


@dataclass(frozen=True)
class ValidationSummary:
    """Auditable summary produced by transaction validation."""

    rows: int
    customers: int
    minimum_date: str
    maximum_date: str
    missing_customer_rate: float
    duplicate_event_rate: float
    cancellation_rate: float
    negative_price_rate: float

    def to_dict(self) -> dict[str, int | float | str]:
        """Return a JSON-serializable mapping."""
        return asdict(self)


def validate_transactions(frame: pd.DataFrame) -> ValidationSummary:
    """Validate schema, date range, prices, identifiers, and duplicate events."""
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if frame.empty:
        raise DataValidationError("Transaction dataset is empty")
    if not pd.api.types.is_datetime64_any_dtype(frame["invoice_date"]):
        raise DataValidationError("invoice_date must be a pandas datetime column")
    if frame["invoice_date"].isna().any():
        raise DataValidationError("invoice_date contains unparseable values")
    if frame["unit_price"].isna().any() or frame["quantity"].isna().any():
        raise DataValidationError("quantity and unit_price must be numeric and non-missing")
    negative_price_rate = float(frame["unit_price"].lt(0).mean())
    if negative_price_rate > 0.001:
        raise DataValidationError(
            f"negative unit_price rate {negative_price_rate:.2%} exceeds 0.1%"
        )

    customer_missing = frame["customer_id"].isna() | frame["customer_id"].eq("")
    usable = frame.loc[~customer_missing].copy()
    if usable.empty:
        raise DataValidationError("No transactions have a usable customer identifier")
    duplicate_rate = float(usable["event_id"].duplicated().mean())
    if duplicate_rate > 0.05:
        raise DataValidationError(f"Duplicate event rate {duplicate_rate:.2%} exceeds 5%")

    return ValidationSummary(
        rows=len(usable),
        customers=int(usable["customer_id"].nunique()),
        minimum_date=usable["invoice_date"].min().isoformat(),
        maximum_date=usable["invoice_date"].max().isoformat(),
        missing_customer_rate=float(customer_missing.mean()),
        duplicate_event_rate=duplicate_rate,
        cancellation_rate=float(usable["is_cancellation"].mean()),
        negative_price_rate=negative_price_rate,
    )
