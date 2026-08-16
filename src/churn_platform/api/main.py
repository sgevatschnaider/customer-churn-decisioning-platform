"""FastAPI application for calibrated predictions and economic recommendations."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from churn_platform.api.schemas import (
    BatchDecisionItem,
    BatchDecisionRequest,
    BatchDecisionResponse,
    CustomerFeatures,
    DecisionResponse,
    PredictResponse,
)
from churn_platform.config import load_data_config, project_path
from churn_platform.decisioning.economics import (
    EconomicScenario,
    estimate_margin_at_risk,
    expected_net_value,
    load_economic_scenario,
    validate_horizon_alignment,
)
from churn_platform.decisioning.policy import apply_retention_policy
from churn_platform.models.train import ModelBundle, load_model_bundle


class ModelRuntime:
    """Lazy artifact loader that keeps service startup observable when no model exists."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        configured = model_path or os.getenv("MODEL_PATH", "artifacts/model.joblib")
        self.model_path = project_path(configured)
        self.bundle: ModelBundle | None = None
        self.load_error: str | None = None

    def load(self) -> ModelBundle:
        """Load once, surfacing an actionable service-unavailable error."""
        if self.bundle is not None:
            return self.bundle
        try:
            self.bundle = load_model_bundle(self.model_path)
            self.load_error = None
            return self.bundle
        except Exception as exc:
            self.load_error = str(exc)
            raise HTTPException(
                status_code=503, detail=f"Model artifact unavailable: {exc}"
            ) from exc


def _warnings(features: CustomerFeatures) -> list[str]:
    warnings = []
    if features.purchase_regularity_days is None:
        warnings.append("purchase_regularity_days is missing and will be imputed by the model")
    if features.recency_days > features.customer_tenure_days:
        warnings.append("recency_days exceeds observed tenure; verify feature construction")
    if features.geographic_segment not in {"United Kingdom", "Europe", "Rest of world"}:
        warnings.append("unseen geographic segment will be handled as an unknown category")
    return warnings


def _feature_frame(features: CustomerFeatures) -> pd.DataFrame:
    return pd.DataFrame([features.model_dump()]).drop(
        columns=["customer_id", "observation_history_complete"]
    )


def _operational_eligibility(features: CustomerFeatures) -> tuple[bool, list[str]]:
    policy = load_data_config().eligibility_config()
    reasons = []
    if not features.observation_history_complete:
        reasons.append("incomplete_observation_history")
    if features.recency_days > policy.max_recency_days:
        reasons.append("recency_above_maximum")
    if features.number_of_invoices < policy.minimum_invoices:
        reasons.append("insufficient_invoices")
    return not reasons, reasons


def create_app(
    model_path: str | Path | None = None,
    decisioning_config: str | Path | None = None,
) -> FastAPI:
    """Create an isolated application instance for runtime and tests."""
    application = FastAPI(
        title="Customer Churn Decisioning Platform API",
        version="1.0.1",
        description="Calibrated churn scoring and assumption-based retention decisioning.",
    )
    runtime = ModelRuntime(model_path)
    config_path = decisioning_config or os.getenv("DECISIONING_CONFIG", "configs/decisioning.yaml")

    def scenario() -> EconomicScenario:
        economic_scenario = load_economic_scenario(config_path)
        validate_horizon_alignment(load_data_config().horizon_days, economic_scenario)
        return economic_scenario

    @application.get("/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "live"}

    @application.get("/health/ready", response_model=None)
    def readiness():
        try:
            bundle = runtime.load()
            scenario()
            return {"status": "ready", "model_version": bundle.model_version}
        except Exception as exc:
            detail = runtime.load_error or str(exc) or "runtime configuration unavailable"
            return JSONResponse(status_code=503, content={"status": "degraded", "detail": detail})

    @application.get("/health", response_model=None)
    def health():
        """Compatibility alias for readiness; use the explicit endpoints operationally."""
        return readiness()

    @application.get("/model-info")
    def model_info() -> dict[str, object]:
        bundle = runtime.load()
        return {
            "model_name": bundle.model_name,
            "model_version": bundle.model_version,
            "trained_at_utc": bundle.trained_at_utc,
            "training_period": bundle.training_period,
            "features": bundle.feature_names,
            "test_metrics": bundle.test_metrics,
        }

    @application.post("/predict", response_model=PredictResponse)
    def predict(features: CustomerFeatures) -> PredictResponse:
        bundle = runtime.load()
        probability = float(bundle.estimator.predict_proba(_feature_frame(features))[:, 1][0])
        return PredictResponse(
            customer_id=features.customer_id,
            churn_probability=probability,
            model_version=bundle.model_version,
            scoring_timestamp=datetime.now(UTC).isoformat(),
            warnings=_warnings(features),
        )

    @application.post("/decision", response_model=DecisionResponse)
    def decision(features: CustomerFeatures) -> DecisionResponse:
        prediction = predict(features)
        economic_scenario = scenario()
        margin = float(
            estimate_margin_at_risk(
                np.array([features.monetary_value]), economic_scenario, feature_history_days=180
            )[0]
        )
        net_value = float(
            expected_net_value(
                np.array([prediction.churn_probability]), np.array([margin]), economic_scenario
            )[0]
        )
        operationally_eligible, eligibility_reasons = _operational_eligibility(features)
        economically_eligible = operationally_eligible and net_value > 0
        if eligibility_reasons:
            reason = "Operational eligibility failed: " + ", ".join(eligibility_reasons)
        elif net_value <= 0:
            reason = "Expected net value is not positive under the configured scenario"
        else:
            reason = "Positive expected net value; portfolio ranking is still required"
        return DecisionResponse(
            **prediction.model_dump(),
            margin_at_risk=margin,
            expected_net_value=net_value,
            economic_eligibility=economically_eligible,
            reason=reason,
            economic_scenario=economic_scenario.to_dict(),
            portfolio_selection_notice=(
                "This single-customer result is not a final contact decision. Submit the complete "
                "eligible portfolio to POST /batch-decisions so shared financial and operational "
                "constraints can be enforced."
            ),
        )

    @application.post("/batch-decisions", response_model=BatchDecisionResponse)
    def batch_decisions(request: BatchDecisionRequest) -> BatchDecisionResponse:
        identifiers = [customer.customer_id for customer in request.customers]
        if len(set(identifiers)) != len(identifiers):
            raise HTTPException(status_code=422, detail="customer_id values must be unique")
        bundle = runtime.load()
        economic_scenario = scenario()
        feature_frame = pd.DataFrame([customer.model_dump() for customer in request.customers])
        probabilities = bundle.estimator.predict_proba(
            feature_frame.drop(columns=["customer_id", "observation_history_complete"])
        )[:, 1]
        feature_frame["churn_probability"] = probabilities
        operational = [_operational_eligibility(customer) for customer in request.customers]
        feature_frame["operationally_eligible"] = [item[0] for item in operational]
        feature_frame["operational_reason"] = [",".join(item[1]) for item in operational]
        eligible_scores = feature_frame.loc[feature_frame["operationally_eligible"]].copy()
        if eligible_scores.empty:
            raise HTTPException(
                status_code=422,
                detail="Portfolio contains no operationally eligible active repeat customers",
            )
        ranking, summary = apply_retention_policy(
            eligible_scores,
            economic_scenario,
            policy="expected_value",
            feature_history_days=load_data_config().history_days,
        )
        ranked_by_id = ranking.set_index("customer_id")
        decisions = []
        next_rank = len(ranking) + 1
        for row in feature_frame.itertuples(index=False):
            if row.operationally_eligible:
                ranked = ranked_by_id.loc[row.customer_id]
                decisions.append(
                    BatchDecisionItem(
                        customer_id=row.customer_id,
                        churn_probability=float(ranked["churn_probability"]),
                        margin_at_risk=float(ranked["estimated_margin_at_risk"]),
                        expected_net_value=float(ranked["expected_net_value"]),
                        economic_eligibility=bool(ranked["economic_eligibility"]),
                        recommended_action=str(ranked["recommended_action"]),
                        policy_rank=int(ranked["policy_rank"]),
                        reason=str(ranked["selection_reason"]),
                    )
                )
            else:
                margin = float(
                    estimate_margin_at_risk(
                        np.array([row.monetary_value]),
                        economic_scenario,
                        load_data_config().history_days,
                    )[0]
                )
                net_value = float(
                    expected_net_value(
                        np.array([row.churn_probability]), np.array([margin]), economic_scenario
                    )[0]
                )
                decisions.append(
                    BatchDecisionItem(
                        customer_id=row.customer_id,
                        churn_probability=float(row.churn_probability),
                        margin_at_risk=margin,
                        expected_net_value=net_value,
                        economic_eligibility=False,
                        recommended_action="do_not_contact",
                        policy_rank=next_rank,
                        reason="Operational eligibility failed: " + row.operational_reason,
                    )
                )
                next_rank += 1
        decisions.sort(key=lambda item: item.policy_rank)
        return BatchDecisionResponse(
            model_version=bundle.model_version,
            scoring_timestamp=datetime.now(UTC).isoformat(),
            economic_scenario=economic_scenario.to_dict(),
            binding_constraint=str(summary["binding_constraint"]),
            expected_campaign_cost=float(summary["expected_campaign_cost"]),
            remaining_budget=float(summary["remaining_budget"]),
            budget_utilization_percentage=float(summary["budget_utilization_percentage"]),
            budget_based_contact_capacity=int(summary["budget_based_contact_capacity"]),
            operations_based_contact_capacity=int(summary["operations_based_contact_capacity"]),
            economically_eligible_customers=int(summary["economically_eligible_customers"]),
            actual_selected_customers=int(summary["actual_selected_customers"]),
            expected_value_per_contacted_customer=float(summary["expected_net_value_per_contact"]),
            decisions=decisions,
        )

    return application


app = create_app()
