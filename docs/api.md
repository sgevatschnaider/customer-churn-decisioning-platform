# API Contract

## Run locally

Train a fixture or UCI model first, then start:

```bash
python -m uvicorn churn_platform.api.main:app --host 0.0.0.0 --port 8000
```

The model path defaults to `artifacts/model.joblib`. Override it with `MODEL_PATH`. Override the economic YAML with `DECISIONING_CONFIG`.

## Endpoints

### `GET /health`

Returns `ready` with model version when the artifact can be loaded. Returns `degraded` when the service is alive but the artifact is missing or invalid.

### `GET /model-info`

Returns selected model, version, training timestamp, temporal periods, feature contract, and final test metrics. A missing artifact returns HTTP 503.

### `POST /predict`

Validates a complete point-in-time feature record and returns customer ID, calibrated churn probability, model version, UTC scoring timestamp, and non-blocking warnings. Unknown fields and out-of-range values return HTTP 422.

### `POST /decision`

Returns the prediction plus estimated value at risk, expected net value, action, reason, and the complete economic scenario. The endpoint evaluates one customer; campaign capacity and portfolio ranking are applied by the batch policy stage.

## Request example

```json
{
  "customer_id": "EXAMPLE-001",
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
  "frequency_change_ratio": 0.6
}
```

## Security posture

The API exposes no training, deletion, administrative, shell, or secret-management endpoints. Pydantic forbids extra fields and constrains numeric ranges. Deployment must add standard infrastructure controls such as TLS, authentication, rate limiting, request-size limits, audit logs, and lawful customer-data handling.

