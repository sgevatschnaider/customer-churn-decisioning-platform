# Repository Guidance

## Scope and runtime

- Python support: 3.11 and 3.12 (`requires-python = ">=3.11,<3.13"`).
- Application code lives in `src/churn_platform/`; orchestration is in `dags/`; reusable
  configuration is in `configs/`.
- Install reproducibly with `python -m pip install -r requirements.lock` followed by
  `python -m pip install --no-deps -e .`.

## Required checks

Run `python -m ruff check .`, `python -m ruff format --check .`, and `python -m pytest` before a
pull request. Combined line and branch coverage must remain at or above 80%. Run
`python -m churn_platform.cli pipeline --source fixture` for the credential-free smoke path.

## Data and evidence rules

- Preserve point-in-time feature construction: features end at the cutoff, labels use only the
  future horizon, and temporal partitions must not overlap.
- The UCI run is professional evidence; the deterministic fixture is only for tests and CI.
- Never publish local filesystem paths, secrets, raw customer identifiers, fabricated screenshots,
  or unexecuted results. Public path fields must be repository-relative and use POSIX separators.
- Scenario expected value is not causal profit. Do not describe this repository as a live enterprise
  deployment.
- If model logic or economic assumptions change, regenerate every affected artifact, report, chart,
  and README result from one traceable run. Do not mix UCI and fixture outputs.

## Pull request acceptance

A pull request must explain behavior and evidence, keep the fixture pipeline deterministic, pass
lint/format/tests/coverage, preserve leakage controls, keep tracked files publication-safe, and state
honestly which Docker, Airflow, MLflow, and API checks were actually executed.
