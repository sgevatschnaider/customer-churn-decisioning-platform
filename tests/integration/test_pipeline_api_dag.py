from __future__ import annotations

import importlib

import pandas as pd
from fastapi.testclient import TestClient

from churn_platform.api.main import create_app
from churn_platform.models.train import load_model_bundle


def api_payload() -> dict[str, object]:
    return {
        "customer_id": "SYN_API_001",
        "recency_days": 45,
        "purchase_frequency": 1.2,
        "monetary_value": 1200,
        "average_order_value": 100,
        "customer_tenure_days": 180,
        "number_of_invoices": 12,
        "number_of_unique_products": 20,
        "purchase_regularity_days": 8,
        "recent_purchasing_trend": -0.2,
        "cancellation_rate": 0.05,
        "quantity_purchased": 60,
        "geographic_segment": "Europe",
        "recent_spend": 200,
        "historical_spend": 400,
        "spend_change_ratio": 0.5,
        "recent_invoice_count": 2,
        "historical_invoice_count": 4,
        "frequency_change_ratio": 0.6,
    }


def test_complete_fixture_pipeline(pipeline_result, project_root) -> None:
    assert pipeline_result["source"] == "fixture"
    assert pipeline_result["selected_model"]
    assert pipeline_result["test_metrics"]["roc_auc"] >= 0.5
    assert (project_root / "artifacts" / "retention_targets.csv").exists()
    assert (project_root / "reports" / "business_results.md").exists()
    assert (project_root / "reports" / "monitoring_report.md").exists()
    targets = pd.read_csv(project_root / "artifacts" / "retention_targets.csv")
    assert {"recommended_action", "expected_net_value"}.issubset(targets.columns)
    bundle = load_model_bundle()
    assert bundle.test_metrics["brier_score"] >= 0


def test_api_endpoints_and_validation(pipeline_result) -> None:
    client = TestClient(create_app())
    assert client.get("/health").json()["status"] == "ready"
    model_info = client.get("/model-info")
    assert model_info.status_code == 200
    prediction = client.post("/predict", json=api_payload())
    assert prediction.status_code == 200
    assert 0 <= prediction.json()["churn_probability"] <= 1
    decision = client.post("/decision", json=api_payload())
    assert decision.status_code == 200
    assert decision.json()["recommended_action"] in {"contact", "do_not_contact"}
    invalid = {**api_payload(), "cancellation_rate": 2.0}
    assert client.post("/predict", json=invalid).status_code == 422


def test_degraded_api_and_dag_contract(tmp_path) -> None:
    client = TestClient(create_app(model_path=tmp_path / "missing.joblib"))
    assert client.get("/health").json()["status"] == "degraded"
    assert client.post("/predict", json=api_payload()).status_code == 503
    module = importlib.import_module("dags.churn_pipeline")
    assert len(module.TASK_IDS) == 11
    assert module.TASK_IDS[0] == "download_or_ingest_data"
    assert module.TASK_IDS[-1] == "generate_monitoring_baseline"
