# ADR 006: Separate predictive risk from scenario economics

- Status: Accepted
- Date: 2026-08-15

## Context

High churn probability does not imply high financial priority. Contact budget and customer value must be part of the decision, but these assumptions should not contaminate model fitting.

## Decision

Calculate margin at risk from historical monetary value, gross-margin rate, and economic horizon. Rank positive expected net value using predicted churn, assumed retention probability, contact cost, and expected offer cost. Constrain both maximum fraction and total worst-case spend. Keep all assumptions in YAML.

## Consequences

The value policy can rationally sacrifice churn recall to protect larger margins, as observed in the published run. Changes in economics do not require retraining, but they require sensitivity analysis and business approval.

