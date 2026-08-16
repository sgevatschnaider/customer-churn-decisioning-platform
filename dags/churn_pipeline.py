"""Airflow orchestration for the customer churn decisioning pipeline."""

from __future__ import annotations

import os
from datetime import datetime
from itertools import pairwise
from typing import Any

from churn_platform.pipeline import run_stage

TASK_IDS = [
    "download_or_ingest_data",
    "validate_raw_data",
    "build_customer_snapshots",
    "validate_point_in_time_features",
    "train_candidate_models",
    "evaluate_and_select_model",
    "register_experiment_and_artifacts",
    "score_eligible_customers",
    "apply_retention_policy",
    "generate_business_report",
    "generate_monitoring_baseline",
]


def execute_stage(stage: str, source: str) -> str:
    """Run a reusable package stage while keeping large frames out of XCom."""
    run_stage(stage, source)
    return stage


try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # Allows lightweight CI to validate module structure.
    DAG = None  # type: ignore[assignment,misc]
    PythonOperator = None  # type: ignore[assignment,misc]


dag: Any = None
if DAG is not None and PythonOperator is not None:
    pipeline_source = os.getenv("PIPELINE_SOURCE", "uci")
    with DAG(
        dag_id="customer_churn_decisioning_pipeline",
        description="Point-in-time churn modeling and budget-constrained retention decisioning",
        start_date=datetime(2024, 1, 1),
        schedule=None,
        catchup=False,
        max_active_runs=1,
        tags=["machine-learning", "decisioning", "retention"],
        default_args={"owner": "ml-platform", "retries": 1},
    ) as dag:
        stages = [
            ("download_or_ingest_data", "ingest"),
            ("validate_raw_data", "validate"),
            ("build_customer_snapshots", "features"),
            ("validate_point_in_time_features", "point_in_time"),
            ("train_candidate_models", "train"),
            ("evaluate_and_select_model", "evaluate"),
            ("register_experiment_and_artifacts", "register"),
            ("score_eligible_customers", "score"),
            ("apply_retention_policy", "decision"),
            ("generate_business_report", "report"),
            ("generate_monitoring_baseline", "monitoring"),
        ]
        tasks = [
            PythonOperator(
                task_id=task_id,
                python_callable=execute_stage,
                op_kwargs={"stage": stage, "source": pipeline_source},
            )
            for task_id, stage in stages
        ]
        for upstream, downstream in pairwise(tasks):
            upstream >> downstream
