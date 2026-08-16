FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .
COPY configs ./configs
COPY artifacts ./artifacts

EXPOSE 8000 5000
CMD ["uvicorn", "churn_platform.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

