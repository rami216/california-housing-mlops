![CI](https://github.com/rami216/california-housing-mlops/actions/workflows/ci.yml/badge.svg)

# California Housing — end-to-end ML service

A house price regression model, but the point isn't the model. The point is
everything around it: preprocessing that survives deployment, a real API,
a container, and CI that retrains and verifies the whole thing on every push.

Built with PyTorch and scikit-learn. Dataset is the 1990 California census
block groups (20,640 rows, 8 features), target is median house value.

## Try it

```bash
docker run -p 8000:8000 ghcr.io/rami216/california-housing-mlops:latest
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 8.3252, "HouseAge": 41, "AveRooms": 6.984127,
       "AveBedrms": 1.02381, "Population": 322, "AveOccup": 2.555556,
       "Latitude": 37.88, "Longitude": -122.23}'
```

```json
{
  "predicted_value": 4.1667,
  "predicted_usd": 416666.22,
  "model_version": "0.1.0"
}
```

Interactive docs at http://localhost:8000/docs

## Results

|                  | RMSE   | in dollars |
| ---------------- | ------ | ---------- |
| predict the mean | 1.1448 | $114,479   |
| this model       | 0.5191 | $51,913    |

MAE $35,381, R² 0.79.

Broken down further: RMSE is 0.48 on normal rows but 1.03 on the 4.5% of rows
where the target is capped at $500,001. The 1990 census truncated everything
above that, so a $900k block group and a $500k one look identical in the data.
Roughly a third of the total error comes from rows the model can't get right.

## What's in here

**Preprocessing** — a scikit-learn Pipeline with two custom transformers.
`FeatureEngineer` adds distance-to-nearest-city (SF or LA), which correlates
-0.44 with price where raw latitude and longitude manage -0.14 and -0.05.
`QuantileClipper` caps the extreme values — `AveOccup` has a median of 2.8 and
a max of 1243, which would have wrecked the scaler.

The pipeline is fitted on training data only and saved as an artifact. The API
loads it alongside the model weights, so inference goes through the identical
transformation training did.

**Model** — a small MLP, 10 → 64 → 32 → 1, 2,817 parameters. Adam, MSE, early
stopping on validation loss with the best weights restored.

**API** — FastAPI. Model loads once at startup, not per request. Pydantic
validates inputs against California's actual coordinate bounds, so a request
with latitude 99 gets a 422 instead of a confident nonsense prediction.

**Container** — CPU-only torch, non-root user, healthcheck. Model path comes
from an environment variable so the same image runs anywhere.

**CI** — on every push, a clean Ubuntu runner installs the package, prepares
the data, trains the model, runs the tests, builds the image, starts the
container and hits it with a real HTTP request. Only then does it publish to
GHCR. Every image is tagged with its commit SHA, and every model file records
the commit that produced it.

## Monitoring

Three endpoints beyond `/predict`:

- `/health` — liveness, used by the Docker healthcheck
- `/metrics` — Prometheus format: request counts, latency histogram,
  prediction distribution, per-feature PSI, extrapolation count
- `/drift` — PSI of the last 500 requests against the training distribution

Every prediction logs one JSON line with all eight inputs, the output, the
latency, the model version, and a request ID.

**Drift detection** uses Population Stability Index against bin edges saved
from the training split. Sending 500 rows from the training distribution gives
PSI around 0.02 per feature. Sending a synthetic "wealthy coastal" population
gives 10.6 on `MedInc` — while every request still returns 200 OK with normal
latency and clean logs.

That gap is the point. In that test the model also returned predictions up to
6.40, above the 5.00001 ceiling of everything it was trained on. Nothing in the
HTTP layer, the logs, or the health check indicates a problem. Stale models
don't crash, they quietly stop being right.

PSI has a noise floor of roughly `(bins - 1) / N`, so with 10 bins the monitor
refuses to report below 200 samples — otherwise the standard 0.1 threshold
produces constant false alarms.

## Layout

```
src/ca_housing/     features, model, training, prediction, API
scripts/            explore, prepare, train, evaluate
tests/              pytest
Dockerfile
.github/workflows/  CI
```

## Running it yourself

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/prepare_data.py
python scripts/train_model.py
python scripts/evaluate.py
pytest

python -m uvicorn ca_housing.api:app --reload
```

Note: `numpy<2` is pinned. torch 2.2.2 is the last macOS x86_64 build and it
predates NumPy 2 support.

## Notes

Gradient boosting would beat this on tabular data — around 0.45 RMSE versus
0.52. PyTorch was the deliberate choice here because the goal was the
deployment chain, and a real training loop with real artifacts exercises more
of it than `model.fit()` does.
