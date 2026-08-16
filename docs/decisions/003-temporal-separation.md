# ADR 003: Use non-overlapping temporal partition windows

- Status: Accepted
- Date: 2026-08-15

## Context

Random row splits would mix later customer behavior into model selection and produce optimistic estimates. Even cutoff-based splits can leak when a training label horizon extends into the validation feature date.

## Decision

Use rolling training cutoffs followed by one validation cutoff and one final test cutoff. Require every earlier label window to end before the next partition cutoff. Select and calibrate on validation; evaluate test once.

## Consequences

The estimate better represents forward scoring and explicitly controls label-window overlap. Fewer cutoffs are available than with random splitting, so uncertainty remains and future backtests should add longer histories.

