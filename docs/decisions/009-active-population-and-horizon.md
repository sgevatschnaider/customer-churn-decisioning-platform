# ADR 009: Define an active repeat-buyer population and align horizons

- Status: Accepted
- Date: 2026-08-16

## Context

An unrestricted snapshot can classify already-inactive customers rather than active customers at
risk. Separately, valuing a 45-day churn probability over 90 days would imply an unsupported
probability transformation.

## Decision

Restrict every model and campaign cohort to customers with at least two positive-purchase invoices,
recency no greater than 81 days, and complete 180-day observation and 45-day label windows. Derive
81 days as the ceiling of the 90th percentile of repeat-purchase intervals observed no later than
the final training cutoff. Set the economic horizon to the same 45 days as the churn target.

## Consequences

The model addresses a smaller, operationally actionable population and all reported results must be
regenerated. Population exclusions become first-class audit artifacts. A 90-day campaign valuation
would require a separately trained 90-day probability model or survival analysis; a 45-day
probability will not be extrapolated.

