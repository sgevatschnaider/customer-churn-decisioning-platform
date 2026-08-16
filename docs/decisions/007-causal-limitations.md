# ADR 007: Treat retention probability as non-causal

- Status: Accepted
- Date: 2026-08-15

## Context

Online Retail contains no randomized treatment, outreach, offer acceptance, or counterfactual outcome. Estimating the effect of contact from this source is impossible.

## Decision

Name retention probability an assumed scenario parameter everywhere. Never describe expected net value as identified uplift or guaranteed profit. Include this limitation in README, model card, business report, API scenario output, and ADRs.

## Consequences

Scenario comparisons support planning but cannot authorize a campaign by themselves. A randomized retention experiment is required before causal optimization or uplift modeling.

