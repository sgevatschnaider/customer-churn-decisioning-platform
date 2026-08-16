# Model Card

## Purpose

Prioritize customers for retention review using calibrated churn risk and a separate economic scenario layer. The system is production-oriented, not a claim of live enterprise deployment.

## Population and data

- Data source: UCI Online Retail
- Snapshot rows: 11,213
- Distinct source customer identifiers: 3,749
- Feature cutoffs: 2011-06-01 to 2011-10-17
- Target: no positive purchase in the 45 days strictly after a snapshot cutoff.
- Features: recency, frequency, monetary value, order value, tenure, invoice/product counts, regularity, recent trends, returns, quantity, geography, and window-over-window changes.

## Validation strategy

Models were fitted on historical snapshots, selected only on a later validation cutoff, calibrated on that validation period, and evaluated once on the final temporal test cutoff. Label windows end before the next partition cutoff.

## Selected model

- Model: hist_gradient_boosting
- Execution source version: `a88e9a01dcdcbc706a4c71ee918f466dbec17bdf` (local commit recorded by the actual MLflow run)
- Published code-equivalent commit: `b9c49045b5823553a2ab88774129384151323b11` (same source and tests in the public GitHub history)
- Trained at: 2026-08-15T23:49:15.264421+00:00

## Test metrics

- ROC-AUC: 0.7148
- PR-AUC / average precision: 0.7026
- Brier score: 0.2128
- Precision at budget: 0.7638
- Recall at budget: 0.2160
- Lift at budget: 1.4423
- F1 at budget: 0.3367

The operating confusion matrix is stored in `artifacts/metrics.json`; its positive class is the budget-constrained contact policy, not a universal 0.5 classification threshold.

## Limitations and risks

The source has no retention treatment, campaign response, marketing consent, customer acquisition cost, or causal outcome. The configured retention probability is an assumption, not an identified treatment effect. Expected net value is therefore scenario analysis and must not be presented as guaranteed incremental profit.

Customer IDs are operational pseudonyms, not identities. Geographic segment may proxy for protected or commercially sensitive attributes; it requires fairness review and lawful-use assessment before deployment. The model should not be used for pricing, credit, eligibility, or adverse customer treatment.

## Conditions for non-use

Do not use with incomplete horizons, schema failures, unresolved drift alerts, a materially different customer population, unreviewed campaign costs, or where outreach lacks a lawful basis.

## Monitoring

Validate schema and drift for every scoring batch, review probability distributions weekly, and assess calibration and ranking performance after labels mature. Human owners decide whether alerts justify pausing, investigation, or retraining.
