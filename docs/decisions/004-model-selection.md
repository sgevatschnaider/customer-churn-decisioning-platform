# ADR 004: Compare interpretable and nonlinear scikit-learn pipelines

- Status: Accepted
- Date: 2026-08-15

## Context

The portfolio must demonstrate an interpretable baseline and nonlinear capacity without adding fragile native dependencies.

## Decision

Compare a business heuristic, regularized class-weighted logistic regression, and HistGradientBoosting. Put imputation, encoding, and scaling inside scikit-learn pipelines. Use controlled seeds and external YAML parameters.

## Consequences

Logistic regression provides a strong interpretable reference; HistGradientBoosting captures interactions and won the published validation. Avoiding XGBoost/LightGBM reduces build risk and image size while preserving a credible nonlinear candidate.

