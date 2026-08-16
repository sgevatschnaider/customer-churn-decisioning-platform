# Business Results

## Executive summary

This report is generated from an actual **UCI Online Retail** pipeline run. Under a 15% contact limit, the value-aware policy selected 453 customers, captured 12.0% of observed churners, and produced scenario expected net value of GBP 21,156.58.

## Budget and recommended policy

- Scenario budget: GBP 12,000.00
- Contact cost: GBP 2.00
- Offer cost: GBP 12.00
- Assumed retention probability: 25%
- Maximum contact fraction: 15%
- Recommendation: use positive expected net value ranking as a review queue, subject to consent, operational capacity, and an experimental campaign design.

## Policy comparison

| policy            |   customers_contacted |   recall_at_budget |   precision_at_budget |   lift_at_budget | expected_net_value   | scenario_realized_net_value   |
|:------------------|----------------------:|-------------------:|----------------------:|-----------------:|:---------------------|:------------------------------|
| random            |                   453 |              0.149 |                 0.528 |            0.997 | 5,310.80             | 3,857.93                      |
| churn_probability |                   453 |              0.216 |                 0.764 |            1.442 | 4,143.79             | 4,063.72                      |
| expected_value    |                   453 |              0.12  |                 0.424 |            0.8   | 21,156.58            | 14,588.61                     |

The random benchmark reports the mean across 200 deterministic seeded policy draws. `scenario_realized_net_value` uses observed churn labels and the same assumed retention rate; it is still not causal profit.

## Sensitivity

Across 27 unique configurations, expected net value ranged from GBP 4,963.88 to GBP 45,031.17. The analysis varies retention probability, offer cost, and gross-margin scaling.

## Risks and next steps

The source contains no treatment assignment or campaign response, so retention probability is a configurable scenario input rather than an estimated causal effect. A real next step is a randomized controlled retention experiment, followed by uplift modeling and segment-level fairness, consent, deliverability, and capacity checks.
