# Leakage Prevention

## Temporal observation unit

An observation is a `(customer_id, cutoff_date)` pair. Every feature value is computed from events in `(cutoff − 180 days, cutoff]`. The target uses only positive purchases in `(cutoff, cutoff + 45 days]`. A purchase exactly at the cutoff can be a feature event and cannot be a label event.

## Label definition

`churn = 1` when the customer has no positive, non-cancellation purchase during the complete future horizon. Returns and cancellations do not count as retained purchasing activity. A snapshot is rejected if the dataset does not cover the full label window.

## Temporal partitions

| Partition | Cutoff(s) | Latest label date |
|---|---|---|
| Train | 2011-06-01, 2011-07-17 | 2011-08-31 |
| Validation | 2011-09-01 | 2011-10-16 |
| Test | 2011-10-17 | 2011-12-01 |

The latest training label date is earlier than the validation cutoff. The validation label date is earlier than the test cutoff. This prevents future outcomes used to label one partition from becoming contemporaneous features in the next partition.

## Model-selection protocol

Candidate pipelines fit only the training snapshots. Hyperparameters and model family are selected on validation PR-AUC with Brier score as a tie-breaker. The selected fitted classifier is sigmoid-calibrated on validation. The final test cohort is scored once and is never used for selection, tuning, or calibration. Permutation importance uses the final test only for post-selection explanation.

## Event lineage controls

Each normalized line receives a stable event ID from invoice, product, date, and customer fields. Exact source duplicates are reported during validation and deduplicated before feature aggregation. For each cutoff, automated assertions reconstruct feature and label event sets and require an empty intersection.

The test suite verifies:

- `feature_max_event_date <= cutoff_date`;
- observed `label_min_event_date > cutoff_date`;
- label dates lie inside the future horizon;
- feature and label event IDs are disjoint;
- horizons are fully observable;
- train, validation, and test windows are temporally isolated;
- intentionally tampered lineage raises `PointInTimeError`.

## Known boundary

The same customer may appear at multiple historical cutoffs. This is intentional rolling-origin supervision, not row-level random resampling. Statistical dependence across a customer's training snapshots is possible; no snapshot from the final temporal test is used during fitting.

