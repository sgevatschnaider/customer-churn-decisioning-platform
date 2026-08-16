FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHURN_PLATFORM_PROJECT_ROOT=/app

WORKDIR /app
COPY pyproject.toml requirements.lock README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.lock \
    && pip install --no-cache-dir --no-deps .
COPY configs ./configs
COPY data/fixtures ./data/fixtures
RUN mkdir -p artifacts reports/figures data/raw data/interim data/processed

EXPOSE 8000 5000
CMD ["uvicorn", "churn_platform.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
