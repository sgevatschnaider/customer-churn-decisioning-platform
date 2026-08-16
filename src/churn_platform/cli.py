"""Command-line interface for reproducible local operation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from churn_platform.config import load_data_config
from churn_platform.data.fixtures import generate_fixture
from churn_platform.logging import configure_logging
from churn_platform.pipeline import run_pipeline, run_stage


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(prog="churn-platform")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser("generate-fixture", help="Generate deterministic CI data")
    fixture.add_argument("--customers", type=int, default=120)
    pipeline = subparsers.add_parser("pipeline", help="Run the complete pipeline")
    pipeline.add_argument("--source", choices=("fixture", "uci"), default="fixture")
    stage = subparsers.add_parser("stage", help="Run one idempotent pipeline stage")
    stage.add_argument(
        "name",
        choices=(
            "ingest",
            "validate",
            "features",
            "point_in_time",
            "train",
            "evaluate",
            "register",
            "score",
            "decision",
            "report",
            "monitoring",
        ),
    )
    stage.add_argument("--source", choices=("fixture", "uci"), default="fixture")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Execute a CLI command and return a process status code."""
    configure_logging()
    options = build_parser().parse_args(arguments)
    if options.command == "generate-fixture":
        data_config = load_data_config()
        frame = generate_fixture(data_config.fixture_path, customers=options.customers)
        print(json.dumps({"rows": len(frame), "path": data_config.fixture_path}))
        return 0
    if options.command == "pipeline":
        print(json.dumps(run_pipeline(options.source), indent=2, default=str))
        return 0
    result = run_stage(options.name, options.source)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
