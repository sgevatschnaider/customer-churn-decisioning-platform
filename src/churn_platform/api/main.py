"""FastAPI application for calibrated predictions and economic recommendations."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from churn_platform.api.schemas import CustomerFeatures, DecisionResponse, PredictResponse
from churn_platform.config import project_path
from churn_platform.decisioning.economics import (
    EconomicScenario,
    estimate_margin_at_risk,
    expected_net_value,
    load_economic_scenario,
)
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
    return pd.DataFrame([features.model_dump()]).drop(columns="customer_id")


def create_app(
    model_path: str | Path | None = None,
    decisioning_config: str | Path | None = None,
) -> FastAPI:
    """Create an isolated application instance for runtime and tests."""
    application = FastAPI(
        title="Customer Churn Decisioning Platform API",
        version="1.0.0",
        description="Calibrated churn scoring and assumption-based retention decisioning.",
    )
    runtime = ModelRuntime(model_path)
    config_path = decisioning_config or os.getenv("DECISIONING_CONFIG", "configs/decisioning.yaml")

    def scenario() -> EconomicScenario:
        return load_economic_scenario(config_path)

    @application.get("/health")
    def health() -> dict[str, str]:
        try:
            bundle = runtime.load()
            return {"status": "ready", "model_version": bundle.model_version}
        except HTTPException:
            return {"status": "degraded", "detail": runtime.load_error or "model unavailable"}

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
        value_at_risk = prediction.churn_probability * margin
        net_value = float(
            expected_net_value(
                np.array([prediction.churn_probability]), np.array([margin]), economic_scenario
            )[0]
        )
        action = "contact" if net_value > 0 else "do_not_contact"
        reason = (
            "Positive expected net value under the configured scenario"
            if action == "contact"
            else "Expected net value is not positive under the configured scenario"
        )
        return DecisionResponse(
            **prediction.model_dump(),
            estimated_value_at_risk=value_at_risk,
            expected_net_value=net_value,
            recommended_action=action,
            reason=reason,
            economic_scenario=economic_scenario.to_dict(),
        )

    return application


app = create_app()
