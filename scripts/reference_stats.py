import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

RAW_COLUMNS = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
               "Population", "AveOccup", "Latitude", "Longitude"]
N_BINS = 10
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

df = fetch_california_housing(as_frame=True).frame
X = df.drop(columns=["MedHouseVal"])

# same split as training — reference must be the TRAIN portion only
X_rest, _ = train_test_split(X, test_size=0.15, random_state=SEED)
X_train, _ = train_test_split(X_rest, test_size=0.1765, random_state=SEED)

reference = {}
for col in RAW_COLUMNS:
    values = X_train[col].values
    edges = np.quantile(values, np.linspace(0, 1, N_BINS + 1))
    counts, _ = np.histogram(values, bins=edges)

    reference[col] = {
        "bin_edges": edges.tolist(),
        "bin_pct": (counts / len(values)).tolist(),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }

out = {"n_train_rows": len(X_train), "n_bins": N_BINS, "features": reference}

with open(MODELS / "reference_stats.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"reference stats from {len(X_train)} training rows")
for col in RAW_COLUMNS:
    r = reference[col]
    print(f"  {col:12s} mean={r['mean']:9.3f}  std={r['std']:9.3f}")