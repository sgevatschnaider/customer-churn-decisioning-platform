# Customer Eligibility Policy

## Decision objective

The model is intended to prioritize customers who are still operationally active but may stop
purchasing. It must not learn the easier, less useful distinction between active customers and
customers who were already inactive at the cutoff.

Every training, validation, test, monitoring, batch-scoring, and retention-policy row must satisfy
all four criteria:

1. `recency_days <= 81`;
2. at least two distinct positive-purchase invoices in the 180-day feature history;
3. the complete 180-day observation window is present in the source data;
4. the complete 45-day future label window is present in the source data.

The reusable implementation is in `src/churn_platform/features/eligibility.py`; the executable
configuration is in `configs/data.yaml`. Eligibility is applied before any partition enters model
training, evaluation, monitoring, or targeting. Persisted snapshots retain `is_eligible` and
`eligibility_reason` lineage fields, and downstream stages fail closed if an ineligible row appears.

## Training-only threshold derivation

The maximum recency was selected without consulting validation or test performance. Positive
purchase invoices dated on or before the final training cutoff, 2011-07-17, were ordered within
customer. The distribution comprised 6,384 repeat-purchase intervals from 3,051 customers; 1,702
customers had at least two invoices. The observed 90th percentile interval was 80.091 days, rounded
up to the operational ceiling of **81 days**.

The 90th percentile was chosen as a conservative activity boundary: it retains the large majority
of observed repeat-purchase cycles while excluding exceptionally stale accounts. The choice is a
business-operating rule, not a tuned predictive hyperparameter. It should be reviewed if purchase
cadence, seasonality, geography, or product mix changes materially.

`minimum_invoices: 2` makes repeat behavior observable. With one invoice, regularity, trend, and
cadence features cannot describe an established purchasing relationship.

The pipeline recomputes the training-only percentile for a UCI run and stops if it does not equal
the configured threshold. Fixture CI uses the documented UCI threshold; it does not retune policy
from synthetic data.

## Audit outputs

`artifacts/eligibility_report.json` records overall and cutoff-level population counts, churn
prevalence among eligible observations, every criterion failure count, and mutually exclusive
primary exclusion reasons. Because a row may fail more than one criterion, criterion failure counts
can exceed the number of excluded rows.

Automated tests prove that stale or insufficient-history customers are excluded and that downstream
point-in-time validation rejects any ineligible snapshot. Batch API records that fail operational
eligibility are returned as non-selected with an explicit reason and are never inserted into the
portfolio ranking.

## Monitoring and human review

Monitoring compares eligible training and eligible scoring cohorts only. A sudden change in the
share or reasons of exclusions is itself an operational diagnostic and should be reviewed alongside
drift. Eligibility changes require a new training-only cadence analysis, documentation, model
retraining, and validation; they must not be chosen from final-test outcomes.

