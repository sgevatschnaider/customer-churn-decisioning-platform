# ADR 006: Separate predictive risk from scenario economics

- Status: Accepted
- Date: 2026-08-15

## Context

High churn probability does not imply high financial priority. Contact budget and customer value must be part of the decision, but these assumptions should not contaminate model fitting.

## Decision

Calculate margin at risk from historical monetary value, gross-margin rate, and the aligned 45-day
economic horizon. Keep campaign effectiveness separate from offer take-up. Rank positive expected
net value using predicted churn, an assumed incremental retention effect, contact cost, offer
acceptance probability, and offer cost if accepted. Constrain selection by expected financial spend,
maximum contact fraction, and positive expected value. Keep every assumption in YAML and report
which constraint binds.

## Consequences

The value policy can rationally sacrifice churn recall to protect larger margins. Changes in
economics do not require retraining when the prediction horizon is unchanged, but they require
sensitivity analysis and business approval. A different economic horizon requires a compatible
probability model or survival analysis.
