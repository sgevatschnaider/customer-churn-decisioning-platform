# Model Card

## Purpose

Prioritize customers for retention review using calibrated churn risk and a separate economic scenario layer. The system is production-oriented, not a claim of live enterprise deployment.

## Population and data

- Data source: UCI Online Retail
- Snapshot rows: 5,154
- Distinct source customer identifiers: 2,014
- Customers before eligibility across cutoffs: 11213
- Eligible active-repeat observations: 5154
- Excluded observations: 6059
- Final test customers before eligibility: 3025
- Final eligible test customers: 1433
- Feature cutoffs: 2011-06-01 to 2011-10-17
- Target: no positive purchase in the 45 days strictly after a snapshot cutoff.
- Features: recency, frequency, monetary value, order value, tenure, invoice/product counts, regularity, recent trends, returns, quantity, geography, and window-over-window changes.

## Validation strategy

Models were fitted on historical snapshots, selected only on a later validation cutoff, calibrated on that validation period, and evaluated once on the final temporal test cutoff. Label windows end before the next partition cutoff.

## Selected model

- Model: logistic_regression_c_10
- Version: `4e6d02e7fecc7bcf93b7dae704dc6405ec02bdda`
- Exact public source commit: `4e6d02e7fecc7bcf93b7dae704dc6405ec02bdda`
- Execution timestamp: 2026-08-16T20:47:44.695505+00:00
- Trained at: 2026-08-16T20:47:48.677557+00:00
- MLflow tracking backend: local-file-store (`<local-mlruns>`)
- Dataset SHA-256: `f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b`
- Dependency lock identifier: `b23220ccf255f85efac4439e1c3468252515195a0ca3ff546a463d23fa4bdd95`
- Python: 3.12.13
- Configuration hashes: `{'data.yaml': '6811e2705d4a299feb097eb520c767317f020b02cb1a05986d38a7c72f45cd26', 'decisioning.yaml': '418a951557596ef4b919052c3adc964bb3d493c10705e11767b5daf250f4bfde', 'model.yaml': '2c71c5ee8111de6a43ac33b89f9a44bbdae455188b98e3182b18d7070d1a9a12'}`

## Test metrics

- ROC-AUC: 0.7058
- PR-AUC / average precision: 0.5412
- Brier score: 0.2088
- Precision at budget: 0.5561
- Recall at budget: 0.2302
- Lift at budget: 1.5413
- F1 at budget: 0.3256

The operating confusion matrix is stored in `artifacts/metrics.json`; its positive class is the budget-constrained contact policy, not a universal 0.5 classification threshold.

## Limitations and risks

The source has no retention treatment, campaign response, marketing consent, customer acquisition cost, or causal outcome. The incremental-retention effect and offer acceptance probability are separate scenario assumptions, not an identified treatment effect. Expected net value is therefore scenario analysis and must not be presented as guaranteed incremental profit.

Customer IDs are operational pseudonyms, not identities. Geographic segment may proxy for protected or commercially sensitive attributes; it requires fairness review and lawful-use assessment before deployment. The model should not be used for pricing, credit, eligibility, or adverse customer treatment.

## Conditions for non-use

Do not use with incomplete horizons, schema failures, unresolved drift alerts, a materially different customer population, unreviewed campaign costs, or where outreach lacks a lawful basis.

## Monitoring

Validate schema and drift for every scoring batch, review probability distributions weekly, and assess calibration and ranking performance after labels mature. Human owners decide whether alerts justify pausing, investigation, or retraining.
