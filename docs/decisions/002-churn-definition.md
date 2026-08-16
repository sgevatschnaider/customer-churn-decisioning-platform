# ADR 002: Define inactivity churn with complete future horizons

- Status: Accepted
- Date: 2026-08-15

## Context

The retailer has purchases and returns, not subscription cancellation records. A target must be useful and computable without using future information in predictors.

## Decision

Define churn as no positive, non-cancellation purchase during the 45 days strictly after a cutoff. Build features from the preceding 180 days, including the cutoff. Reject cutoffs whose future horizon is incomplete.

## Consequences

The target is reproducible and point-in-time valid. It represents inactivity over a chosen commercial horizon, not customer intent or permanent attrition. Horizon changes create a different model target and require revalidation.

