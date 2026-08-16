# API Contract

## Run locally

Train a fixture or UCI model first, then start:

```bash
python -m uvicorn churn_platform.api.main:app --host 0.0.0.0 --port 8000
```

The model path defaults to `artifacts/model.joblib`. Override it with `MODEL_PATH`. Override the economic YAML with `DECISIONING_CONFIG`.

## Endpoints

### `GET /health/live`

Returns HTTP 200 with `live` whenever the process is running. It does not imply that prediction is
available.

### `GET /health/ready`

Returns HTTP 200 only when both the serialized model and economic configuration load successfully.
It returns HTTP 503 with a degraded status when the service cannot serve a valid prediction.

### `GET /health`

Compatibility alias for readiness. New operational integrations should use the explicit live and
ready endpoints.

### `GET /model-info`

Returns selected model, version, training timestamp, temporal periods, feature contract, and final test metrics. A missing artifact returns HTTP 503.

### `POST /predict`

Validates a complete point-in-time feature record and returns customer ID, calibrated churn probability, model version, UTC scoring timestamp, and non-blocking warnings. Unknown fields and out-of-range values return HTTP 422.

### `POST /decision`

Returns churn probability, margin at risk, expected net value, economic eligibility, reason, the
complete scenario, and an explicit portfolio-selection notice. This endpoint deliberately does not
return a final `contact` action: one customer cannot be ranked against a campaign portfolio or apply
shared financial and operational capacity.

This is a breaking correction to the earlier response contract. Clients that need a final action
must migrate to `POST /batch-decisions`.

### `POST /batch-decisions`

Accepts `{"customers": [...]}` with between 1 and 1,000 complete feature records. The optional
`observation_history_complete` flag defaults to true and must reflect an upstream point-in-time
completeness check. Customer IDs must
be unique within the request. The endpoint:

- excludes operationally ineligible records from ranking;
- scores the remaining portfolio with the same model contract as `/predict`;
- ranks positive expected net value;
- enforces expected financial spend and maximum-contact capacity;
- returns selected and non-selected records with reasons;
- reports budget capacity, operations capacity, economically eligible count, actual selection,
  binding constraint, expected campaign cost, remaining budget, and utilization.

Requests larger than 1,000 records or containing duplicate IDs return HTTP 422. A portfolio with no
operationally eligible customer returns HTTP 422.

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

## Batch request example

Use the complete feature record above for every element:

```bash
curl -X POST http://localhost:8000/batch-decisions \
  -H "Content-Type: application/json" \
  --data-binary @portfolio.json
```

```json
{
  "customers": [
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
  ]
}
```

## Security posture

The API exposes no training, deletion, administrative, shell, or secret-management endpoints.
Pydantic forbids extra fields, constrains numeric ranges, caps batch size, and rejects duplicate
portfolio identifiers. Deployment must add standard infrastructure controls such as TLS,
authentication, rate limiting, byte-size limits, audit logs, and lawful customer-data handling.
