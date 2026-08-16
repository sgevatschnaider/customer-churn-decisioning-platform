PYTHON ?= python
SOURCE ?= fixture

.PHONY: install data validate features train evaluate decision report pipeline api test lint format monitoring docker-up docker-down ci

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) -m churn_platform.cli stage ingest --source uci

validate:
	$(PYTHON) -m churn_platform.cli stage validate --source $(SOURCE)

features:
	$(PYTHON) -m churn_platform.cli stage features --source $(SOURCE)

train:
	$(PYTHON) -m churn_platform.cli stage train --source $(SOURCE)

evaluate:
	$(PYTHON) -m churn_platform.cli stage evaluate --source $(SOURCE)

decision:
	$(PYTHON) -m churn_platform.cli stage score --source $(SOURCE)
	$(PYTHON) -m churn_platform.cli stage decision --source $(SOURCE)

report:
	$(PYTHON) -m churn_platform.cli stage report --source $(SOURCE)

pipeline:
	$(PYTHON) -m churn_platform.cli pipeline --source $(SOURCE)

api:
	$(PYTHON) -m uvicorn churn_platform.api.main:app --reload --port 8000

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

monitoring:
	$(PYTHON) -m churn_platform.cli stage monitoring --source $(SOURCE)

docker-up:
	docker compose --profile api --profile tracking up --build -d

docker-down:
	docker compose --profile api --profile tracking --profile orchestration down

ci:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m pytest
	$(PYTHON) -m churn_platform.cli pipeline --source fixture

