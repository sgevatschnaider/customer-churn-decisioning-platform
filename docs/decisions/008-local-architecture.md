# ADR 008: Use a local-first modular architecture

- Status: Accepted
- Date: 2026-08-15

## Context

The repository must be runnable by a reviewer without cloud accounts while still demonstrating production-style boundaries and orchestration.

## Decision

Make reusable Python functions the source of truth. Provide a CLI as the primary execution path, MLflow local tracking, an Airflow DAG that calls those functions, FastAPI for serving, and Docker Compose profiles backed by PostgreSQL. Use file artifacts locally and keep heavy/generated state ignored.

## Consequences

A clone can execute the entire fixture pipeline without external credentials. The architecture is portable but does not simulate a managed registry, warehouse, secrets manager, or high-availability serving layer.

