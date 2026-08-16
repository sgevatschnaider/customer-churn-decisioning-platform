# ADR 005: Select with ranking, calibration, and budget metrics

- Status: Accepted
- Date: 2026-08-15

## Context

ROC-AUC alone is insensitive to operating capacity and does not measure probability quality or campaign usefulness.

## Decision

Use validation PR-AUC for model selection with Brier score as a tie-breaker. Publish ROC-AUC, PR-AUC, Brier, calibration, precision/recall/F1, recall@budget, precision@budget, lift@budget, confusion matrix, and economic-policy outcomes.

## Consequences

Selection focuses on ranking churners while the final review exposes calibration and business trade-offs. No single metric is treated as sufficient evidence.

