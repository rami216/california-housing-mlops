import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ca_housing.predict import Predictor

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs ONCE at startup, before any request is served
    state["predictor"] = Predictor(MODELS_DIR)
    try:
        with open(MODELS_DIR / "metadata.json") as f:
            state["metadata"] = json.load(f)
    except FileNotFoundError:
        state["metadata"] = {"model_version": "unknown"}
    yield
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
    try:
        value = state["predictor"].predict(house.model_dump())[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction failed: {e}")

    return Prediction(
        predicted_value=round(value, 4),
        predicted_usd=round(value * 100_000, 2),
        model_version=state["metadata"].get("model_version", "unknown"),
    )