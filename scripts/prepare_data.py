import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ca_housing.features import FeatureEngineer, QuantileClipper

SEED = 42
CLIP_COLS = ["AveRooms", "AveBedrms", "AveOccup", "Population", "bedrooms_ratio"]

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# --- load ---
df = fetch_california_housing(as_frame=True).frame
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"].values.astype("float32")

# --- split ---
X_rest, X_test, y_rest, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_rest, y_rest, test_size=0.1765, random_state=SEED)

print(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

# --- preprocess ---
preprocessor = Pipeline([
    ("features", FeatureEngineer()),
    ("clip", QuantileClipper(columns=CLIP_COLS, lower=0.01, upper=0.99)),
    ("scale", StandardScaler()),
])

X_train_p = preprocessor.fit_transform(X_train).astype("float32")   # fit HERE only
X_val_p = preprocessor.transform(X_val).astype("float32")
X_test_p = preprocessor.transform(X_test).astype("float32")

print(f"\nshapes: {X_train_p.shape}  {X_val_p.shape}  {X_test_p.shape}")
print(f"train  mean={X_train_p.mean():6.3f}  std={X_train_p.std():.3f}")
print(f"val    mean={X_val_p.mean():6.3f}  std={X_val_p.std():.3f}")
print(f"test   mean={X_test_p.mean():6.3f}  std={X_test_p.std():.3f}")

# --- save ---
np.savez(DATA / "train.npz", X=X_train_p, y=y_train)
np.savez(DATA / "val.npz", X=X_val_p, y=y_val)
np.savez(DATA / "test.npz", X=X_test_p, y=y_test)
joblib.dump(preprocessor, MODELS / "preprocessor.joblib")

feature_names = list(FeatureEngineer().fit_transform(X_train).columns)
with open(MODELS / "feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)

print("\nsaved -> data/processed/*.npz, models/preprocessor.joblib")