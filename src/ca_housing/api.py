import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from ca_housing.drift import DriftMonitor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ca_housing.predict import Predictor
import time
import uuid

import structlog
from ca_housing.logging_setup import configure_logging
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from fastapi import Response


from ca_housing.metrics import (
    PREDICTIONS, LATENCY, PREDICTION_VALUE, FEATURE_VALUE,
    MODEL_LOADED, MODEL_INFO,
)
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
configure_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_output=os.getenv("LOG_JSON", "true").lower() == "true",
)
log = structlog.get_logger()

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = Predictor(MODELS_DIR)
    try:
        with open(MODELS_DIR / "metadata.json") as f:
            state["metadata"] = json.load(f)
    except FileNotFoundError:
        state["metadata"] = {"model_version": "unknown"}
    try:
        state["drift"] = DriftMonitor(
            MODELS_DIR / "reference_stats.json",
            window=int(os.getenv("DRIFT_WINDOW", "500")),
        )
    except FileNotFoundError:
        state["drift"] = None
        log.warning("drift_monitor_disabled", reason="reference_stats.json not found")
    log.info("startup_complete",
             model_version=state["metadata"].get("model_version"),
             git_commit=state["metadata"].get("git_commit"),
             models_dir=str(MODELS_DIR))
    MODEL_LOADED.set(1)
    MODEL_INFO.info({
        "version": str(state["metadata"].get("model_version", "unknown")),
        "git_commit": str(state["metadata"].get("git_commit", "unknown")),
    })
    yield
    log.info("shutdown")
    MODEL_LOADED.set(0)
    state.clear()

app = FastAPI(title="California Housing API", version="0.1.0", lifespan=lifespan)


class House(BaseModel):
    MedInc: float = Field(..., gt=0, description="median income, $10k units")
    HouseAge: float = Field(..., ge=0, le=100)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., gt=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float = Field(..., ge=32.0, le=42.5)
    Longitude: float = Field(..., ge=-125.0, le=-113.0)


class Prediction(BaseModel):
    predicted_value: float
    predicted_usd: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in state}


@app.get("/metadata")
def metadata():
    return state["metadata"]


@app.post("/predict", response_model=Prediction)
def predict(house: House):
    request_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()
    payload = house.model_dump()

    try:
        value = state["predictor"].predict(payload)[0]
    except Exception as e:
        PREDICTIONS.labels(status="error").inc()
        log.error("prediction_failed",
                  request_id=request_id,
                  error=str(e),
                  error_type=type(e).__name__,
                  **payload)
        raise HTTPException(status_code=500, detail="prediction failed")

    elapsed_ms = (time.perf_counter() - started) * 1000
    elapsed_ms = (time.perf_counter() - started) * 1000
    PREDICTIONS.labels(status="success").inc()
    LATENCY.observe(elapsed_ms / 1000)
    PREDICTION_VALUE.observe(value)
    for name, v in payload.items():
        FEATURE_VALUE.labels(feature=name).observe(v)

    if value > 5.0 or value < 0.15:                    # <-- here
        EXTRAPOLATIONS.inc()
        log.warning("prediction_out_of_training_range",
                    request_id=request_id, prediction=round(value, 4))

    if state.get("drift"):
        state["drift"].record(payload)

    log.info("prediction",
             request_id=request_id,
             prediction=round(value, 4),
             latency_ms=round(elapsed_ms, 2),
             model_version=state["metadata"].get("model_version"),
             **payload)

    return Prediction(
        predicted_value=round(value, 4),
        predicted_usd=round(value * 100_000, 2),
        model_version=state["metadata"].get("model_version", "unknown"),
    )    
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
  
  
@app.get("/drift")
def drift():
    if not state.get("drift"):
        raise HTTPException(status_code=503, detail="drift monitoring not configured")

    report = state["drift"].report()

    if report["status"] == "ok":
        for name, info in report["features"].items():
            DRIFT_PSI.labels(feature=name).set(info["psi"])
        DRIFT_WORST_PSI.set(report["worst_psi"])
        if report["overall_drift"] != "none":
            log.warning("drift_detected",
                        worst_feature=report["worst_feature"],
                        worst_psi=report["worst_psi"],
                        overall=report["overall_drift"])

    return report