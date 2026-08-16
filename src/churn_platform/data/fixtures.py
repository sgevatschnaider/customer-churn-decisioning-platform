"""Deterministic synthetic transactions used exclusively by tests and CI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from churn_platform.config import project_path


def generate_fixture(path: str | Path, seed: int = 42, customers: int = 120) -> pd.DataFrame:
    """Generate a deterministic, non-personal transaction history with churn patterns."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2024-02-15")
    countries = np.array(["United Kingdom", "France", "Germany", "Netherlands"])

    for customer_number in range(customers):
        customer_id = f"SYN{customer_number:04d}"
        country = str(countries[customer_number % len(countries)])
        cadence = int(7 + (customer_number % 31))
        value_scale = float(8 + (customer_number % 17))
        churn_wave = customer_number % 5
        stop_date = end - pd.Timedelta(days=churn_wave * 38)
        first_date = start + pd.Timedelta(days=int(rng.integers(0, 45)))
        purchase_dates = pd.date_range(first_date, stop_date, freq=f"{cadence}D")

        for purchase_index, invoice_date in enumerate(purchase_dates):
            invoice_no = f"S{customer_number:04d}{purchase_index:03d}"
            products = 1 + ((customer_number + purchase_index) % 3)
            for product_index in range(products):
                quantity = int(1 + rng.integers(0, 6))
                unit_price = round(value_scale * float(rng.uniform(0.7, 1.4)), 2)
                rows.append(
                    {
                        "InvoiceNo": invoice_no,
                        "StockCode": f"P{(customer_number + product_index * 13) % 80:04d}",
                        "Description": "Synthetic fixture product",
                        "Quantity": quantity,
                        "InvoiceDate": invoice_date + pd.Timedelta(hours=10 + product_index),
                        "UnitPrice": unit_price,
                        "CustomerID": customer_id,
                        "Country": country,
                    }
                )
            if purchase_index and (customer_number + purchase_index) % 37 == 0:
                rows.append(
                    {
                        "InvoiceNo": f"C{invoice_no}",
                        "StockCode": f"P{customer_number % 80:04d}",
                        "Description": "Synthetic fixture return",
                        "Quantity": -1,
                        "InvoiceDate": invoice_date + pd.Timedelta(days=1),
                        "UnitPrice": round(value_scale, 2),
                        "CustomerID": customer_id,
                        "Country": country,
                    }
                )

    fixture = pd.DataFrame(rows).sort_values("InvoiceDate").reset_index(drop=True)
    destination = project_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_csv(destination, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return fixture
