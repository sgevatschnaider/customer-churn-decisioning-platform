# Monitoring Report

This report compares the scored batch with the training reference. Alerts are diagnostic; they do not trigger automatic retraining or customer actions.

## Batch status

- Status: **ALERT**
- Reference rows: 5416
- Current rows: 3025
- Alerts: 1

## Alerts

| Severity | Column | Issue | Value |
|---|---|---|---|
| warning | customer_tenure_days | psi_drift | 0.34595489080225156 |

## Observed performance

- roc_auc: 0.7148
- average_precision: 0.7026
- brier_score: 0.2128
- recall_at_budget: 0.2160
- precision_at_budget: 0.7638
- lift_at_budget: 1.4423

## Operating guidance

Missing columns and numeric type violations should block scoring. Range, missingness, PSI, and KS alerts require a data owner and model owner to investigate context before retraining, changing thresholds, or stopping a campaign. Any decline in labeled performance requires human review of label maturity, segment mix, calibration, and business costs.
