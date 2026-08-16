# ADR 001: Use UCI Online Retail

- Status: Accepted
- Date: 2026-08-15

## Context

The project requires customer transactions with enough temporal depth to derive churn without private credentials. A portfolio system also needs traceable authorship, terms, and direct acquisition.

## Decision

Use UCI Online Retail (DOI `10.24432/C5BW33`, CC BY 4.0) from the official UCI host. Do not use a Kaggle mirror. Do not commit the workbook. Verify the official ZIP checksum and commit only a clearly synthetic CI fixture and privacy-reduced decision artifacts.

## Consequences

The source supports realistic transactional features and a defensible lineage chain. It has no campaign treatment, margin, consent, or explicit cancellation-of-service outcome, so churn and economics must be derived and carefully limited.

