# Monitoring Report

This report compares the scored batch with the training reference. Alerts are diagnostic; they do not trigger automatic retraining or customer actions.

## Batch status

- Status: **ALERT**
- Reference rows: 2466
- Current rows: 1433
- Alerts: 2

## Alerts

| Severity | Column | Issue | Value |
|---|---|---|---|
| warning | customer_tenure_days | psi_drift | 0.3593672360872435 |
| warning | customer_tenure_days | ks_drift | 0.1629542916957432 |

## Observed performance

- roc_auc: 0.7058
- average_precision: 0.5412
- brier_score: 0.2088
- recall_at_budget: 0.2302
- precision_at_budget: 0.5561
- lift_at_budget: 1.5413

## Operating guidance

Missing columns and numeric type violations should block scoring. Range, missingness, PSI, and KS alerts require a data owner and model owner to investigate context before retraining, changing thresholds, or stopping a campaign. Any decline in labeled performance requires human review of label maturity, segment mix, calibration, and business costs.
