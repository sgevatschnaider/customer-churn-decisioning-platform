# ADR 004: Compare interpretable and nonlinear scikit-learn pipelines

- Status: Accepted
- Date: 2026-08-15

## Context

The portfolio must demonstrate an interpretable baseline and nonlinear capacity without adding fragile native dependencies.

## Decision

Compare a business heuristic, regularized class-weighted logistic regression, and HistGradientBoosting. Put imputation, encoding, and scaling inside scikit-learn pipelines. Use controlled seeds and external YAML parameters.

## Consequences

Logistic regression provides a strong interpretable reference and, with `C=10`, won the corrected published validation on PR-AUC (0.6335 versus 0.5994 for HistGradientBoosting). HistGradientBoosting still supplies a credible nonlinear candidate that can capture interactions. Avoiding XGBoost/LightGBM reduces build risk and image size without removing nonlinear model comparison.
