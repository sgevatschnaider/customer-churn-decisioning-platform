# Operational Evidence Checklist

The source, fixture pipeline, API behavior, DAG contract, and local MLflow file-store run are
verified by automated tests. Docker was not available in the release-authoring environment, so this
repository does not claim a verified container-network MLflow smoke test and does not contain
fabricated interface screenshots.

When Docker is available, capture genuine evidence with the following procedure.

## MLflow server

```bash
docker compose --profile tracking up --build -d mlflow
MLFLOW_TRACKING_URI=http://localhost:5000 \
  python -m churn_platform.cli pipeline --source fixture
curl --fail http://localhost:5000/api/2.0/mlflow/experiments/search
```

Verify that experiment `customer-churn-decisioning` contains a finished run with parameters,
metrics, plots, configuration artifacts, run lineage, and the serialized model. Capture the browser
at `http://localhost:5000` with the run open.

## Airflow

```bash
cp .env.example .env
docker compose --profile orchestration run --rm airflow-init
PIPELINE_SOURCE=fixture docker compose --profile orchestration up --build -d \
  airflow-webserver airflow-scheduler mlflow postgres
```

Verify that `customer_churn_decisioning_pipeline` imports with eleven tasks, trigger a fixture run,
and confirm every task succeeds. Capture the Grid or Graph view at `http://localhost:8080`.

## FastAPI

```bash
docker compose --profile api up --build -d api
curl --fail http://localhost:8000/health/live
curl --fail http://localhost:8000/health/ready
```

Verify that liveness and readiness return HTTP 200, then open `http://localhost:8000/docs`. Exercise
`POST /predict`, `POST /decision`, and `POST /batch-decisions`; capture the genuine OpenAPI page.

## Evidence acceptance

- record the tested public commit and image capture timestamp;
- show service health and a successful fixture execution;
- redact local paths, customer inputs, tokens, and credentials;
- optimize images before adding them under `docs/images/`;
- do not use images as substitutes for machine-verifiable CI evidence.

