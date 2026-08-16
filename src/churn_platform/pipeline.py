"""File-backed pipeline stages shared by CLI, tests, and Airflow."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pandas as pd

from churn_platform.config import load_data_config, load_yaml, project_path
from churn_platform.data.download import download_dataset
from churn_platform.data.fixtures import generate_fixture
from churn_platform.data.ingest import ingest_transactions
from churn_platform.data.validation import validate_transactions
from churn_platform.decisioning.economics import (
    load_economic_scenario,
    validate_horizon_alignment,
)
from churn_platform.decisioning.policy import (
    compare_policies,
    save_decision_outputs,
    sensitivity_analysis,
)
from churn_platform.features.build_features import MODEL_FEATURES
from churn_platform.features.eligibility import (
    assert_eligibility_integrity,
    derive_recency_threshold,
)
from churn_platform.features.snapshots import assert_point_in_time_integrity, build_snapshots
from churn_platform.models.predict import score_customers
from churn_platform.models.train import load_model_bundle, train_and_select
from churn_platform.monitoring.data_drift import (
    build_monitoring_baseline,
    compare_to_baseline,
    save_baseline,
)
from churn_platform.monitoring.performance import (
    evaluate_observed_performance,
    render_monitoring_report,
)
from churn_platform.reporting import render_business_report, render_model_card
from churn_platform.reproducibility import build_run_metadata

LOGGER = logging.getLogger(__name__)


def prepare_data(source: str) -> pd.DataFrame:
    """Acquire/generate and normalize the selected transaction source."""
    config = load_data_config()
    if source == "fixture":
        generate_fixture(config.fixture_path)
        input_path = config.fixture_path
    elif source == "uci":
        download_dataset(
            config.download_url,
            config.raw_zip_path,
            config.raw_xlsx_path,
            expected_sha256=load_yaml("configs/data.yaml").get("expected_sha256"),
        )
        input_path = config.raw_xlsx_path
    else:
        raise ValueError("source must be 'fixture' or 'uci'")
    return ingest_transactions(input_path, config.normalized_path)


def validate_data() -> dict[str, int | float | str]:
    """Validate normalized raw data and persist an auditable summary."""
    config = load_data_config()
    transactions = pd.read_parquet(project_path(config.normalized_path))
    summary = validate_transactions(transactions).to_dict()
    destination = project_path("artifacts/data_validation.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_feature_stage(source: str) -> pd.DataFrame:
    """Build configured point-in-time snapshots."""
    config = load_data_config()
    eligibility = config.eligibility_config()
    transactions = pd.read_parquet(project_path(config.normalized_path))
    if source == "uci":
        threshold_derivation = derive_recency_threshold(
            transactions,
            eligibility.derived_from_training_cutoff,
            eligibility.recency_quantile,
        )
        if threshold_derivation["max_recency_days"] != eligibility.max_recency_days:
            raise ValueError(
                "Configured max_recency_days does not match the training-only derivation: "
                f"{threshold_derivation}"
            )
    else:
        threshold_derivation = {
            "method": "fixed from the documented UCI training-period derivation",
            "max_recency_days": eligibility.max_recency_days,
            "training_cutoff": eligibility.derived_from_training_cutoff,
            "quantile": eligibility.recency_quantile,
        }
    snapshots = build_snapshots(
        transactions,
        config.split_cutoffs(fixture=source == "fixture"),
        config.history_days,
        config.horizon_days,
        eligibility.max_recency_days,
        eligibility.minimum_invoices,
    )
    by_cutoff = snapshots.attrs["eligibility_summary"]
    eligibility_report = {
        "threshold_derivation": threshold_derivation,
        "overall": {
            "total_customers_before_eligibility": int(
                sum(item["total_customers_before_eligibility"] for item in by_cutoff)
            ),
            "eligible_customers": int(sum(item["eligible_customers"] for item in by_cutoff)),
            "excluded_customers": int(sum(item["excluded_customers"] for item in by_cutoff)),
            "eligible_churn_prevalence": float(snapshots["churn"].mean()),
            "criterion_failure_counts": {
                reason: int(sum(item["criterion_failure_counts"][reason] for item in by_cutoff))
                for reason in by_cutoff[0]["criterion_failure_counts"]
            },
            "primary_exclusion_reasons": {
                reason: int(
                    sum(item["primary_exclusion_reasons"].get(reason, 0) for item in by_cutoff)
                )
                for reason in sorted(
                    {reason for item in by_cutoff for reason in item["primary_exclusion_reasons"]}
                )
            },
        },
        "by_cutoff": by_cutoff,
    }
    eligibility_destination = project_path("artifacts/eligibility_report.json")
    eligibility_destination.parent.mkdir(parents=True, exist_ok=True)
    eligibility_destination.write_text(json.dumps(eligibility_report, indent=2), encoding="utf-8")
    destination = project_path(config.snapshots_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(destination, index=False)
    return snapshots


def validate_point_in_time_stage() -> None:
    """Re-run lineage checks on persisted data and features."""
    config = load_data_config()
    snapshots = pd.read_parquet(project_path(config.snapshots_path))
    transactions = pd.read_parquet(project_path(config.normalized_path))
    assert_point_in_time_integrity(snapshots, transactions, config.history_days)
    eligibility = config.eligibility_config()
    assert_eligibility_integrity(
        snapshots, eligibility.max_recency_days, eligibility.minimum_invoices
    )


def train_stage(source: str = "fixture") -> dict[str, Any]:
    """Train candidates, select/calibrate, run final test, and register MLflow artifacts."""
    config = load_data_config()
    snapshots = pd.read_parquet(project_path(config.snapshots_path))
    model_config = load_yaml("configs/model.yaml")
    decisioning_config = load_yaml("configs/decisioning.yaml")
    if source == "uci" and not os.getenv("CHURN_PLATFORM_SOURCE_COMMIT"):
        raise RuntimeError(
            "Full UCI runs require CHURN_PLATFORM_SOURCE_COMMIT to reference the public "
            "executable source commit"
        )
    dataset_path = config.raw_zip_path if source == "uci" else config.fixture_path
    metadata = build_run_metadata(source, project_path(dataset_path))
    bundle, scored_test = train_and_select(
        snapshots,
        model_config,
        decisioning_config=decisioning_config,
        run_metadata=metadata,
    )
    scored_test.to_parquet(project_path("artifacts/scored_test.parquet"), index=False)
    return bundle.test_metrics


def evaluate_stage() -> dict[str, Any]:
    """Return persisted final-test evaluation; training owns the one-time test call."""
    return load_model_bundle().test_metrics


def register_stage() -> dict[str, str]:
    """Verify that a local MLflow run and serialized model were created."""
    path = project_path("artifacts/mlflow_run.json")
    if not path.exists():
        raise FileNotFoundError("MLflow run metadata is missing; run the train stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def score_stage() -> pd.DataFrame:
    """Score the eligible final temporal cohort using the serialized model."""
    config = load_data_config()
    snapshots = pd.read_parquet(project_path(config.snapshots_path))
    test = snapshots.loc[snapshots["split"].eq("test")].copy()
    eligibility = config.eligibility_config()
    assert_eligibility_integrity(test, eligibility.max_recency_days, eligibility.minimum_invoices)
    scored = score_customers(load_model_bundle(), test)
    scored.to_parquet(project_path("artifacts/scored_test.parquet"), index=False)
    return scored


def decision_stage() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare budget-constrained policies and persist the value-aware ranking."""
    data_config = load_data_config()
    scored = pd.read_parquet(project_path("artifacts/scored_test.parquet"))
    scenario = load_economic_scenario()
    validate_horizon_alignment(data_config.horizon_days, scenario)
    comparison, rankings = compare_policies(scored, scenario, data_config.history_days)
    sensitivity = sensitivity_analysis(scored, scenario, data_config.history_days)
    save_decision_outputs(
        rankings,
        comparison,
        sensitivity,
        project_path("artifacts/retention_targets.csv"),
        project_path("reports/figures"),
    )
    return comparison, sensitivity


def report_stage(source: str) -> None:
    """Generate model card and business report from actual artifacts."""
    config = load_data_config()
    snapshots = pd.read_parquet(project_path(config.snapshots_path))
    bundle = load_model_bundle()
    comparison = pd.read_csv(project_path("artifacts/policy_comparison.csv"))
    sensitivity = pd.read_csv(project_path("artifacts/sensitivity_analysis.csv"))
    source_name = (
        "deterministic synthetic CI fixture" if source == "fixture" else "UCI Online Retail"
    )
    eligibility_report = json.loads(
        project_path("artifacts/eligibility_report.json").read_text(encoding="utf-8")
    )
    render_model_card(
        bundle,
        snapshots,
        source_name,
        project_path("reports/model_card.md"),
        eligibility_report=eligibility_report,
    )
    render_business_report(
        comparison,
        sensitivity,
        load_economic_scenario(),
        source_name,
        project_path("reports/business_results.md"),
        eligibility_report=eligibility_report,
    )


def monitoring_stage() -> dict[str, Any]:
    """Create a training baseline and compare the final temporal cohort."""
    config = load_data_config()
    snapshots = pd.read_parquet(project_path(config.snapshots_path))
    bundle = load_model_bundle()
    train = score_customers(bundle, snapshots.loc[snapshots["split"].eq("train")].copy())
    test = score_customers(bundle, snapshots.loc[snapshots["split"].eq("test")].copy())
    baseline = build_monitoring_baseline(train[[*MODEL_FEATURES, "churn_probability"]])
    save_baseline(baseline, project_path("artifacts/monitoring_baseline.json"))
    drift = compare_to_baseline(baseline, test)
    performance = evaluate_observed_performance(
        test, load_yaml("configs/model.yaml")["budget_fraction"]
    )
    render_monitoring_report(drift, performance, project_path("reports/monitoring_report.md"))
    return drift


def run_pipeline(source: str = "fixture") -> dict[str, Any]:
    """Execute the complete local pipeline without requiring Airflow."""
    LOGGER.info("Starting pipeline source=%s", source)
    prepare_data(source)
    validation = validate_data()
    build_feature_stage(source)
    validate_point_in_time_stage()
    train_stage(source)
    evaluate_stage()
    register_stage()
    score_stage()
    comparison, _ = decision_stage()
    report_stage(source)
    drift = monitoring_stage()
    bundle = load_model_bundle()
    result = {
        "source": source,
        "validation": validation,
        "selected_model": bundle.model_name,
        "test_metrics": bundle.test_metrics,
        "value_policy": comparison.loc[comparison["policy"].eq("expected_value")].iloc[0].to_dict(),
        "monitoring_status": drift["status"],
        "eligibility": json.loads(
            project_path("artifacts/eligibility_report.json").read_text(encoding="utf-8")
        )["overall"],
        "run_metadata": bundle.run_metadata,
    }
    project_path("artifacts/pipeline_summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result


def run_stage(stage: str, source: str = "fixture") -> Any:
    """Dispatch one named stage for Airflow and operational recovery."""
    stages = {
        "ingest": lambda: prepare_data(source),
        "validate": validate_data,
        "features": lambda: build_feature_stage(source),
        "point_in_time": validate_point_in_time_stage,
        "train": lambda: train_stage(source),
        "evaluate": evaluate_stage,
        "register": register_stage,
        "score": score_stage,
        "decision": decision_stage,
        "report": lambda: report_stage(source),
        "monitoring": monitoring_stage,
    }
    if stage not in stages:
        raise ValueError(f"Unknown pipeline stage: {stage}")
    return stages[stage]()
