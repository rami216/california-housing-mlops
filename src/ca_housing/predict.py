import json
from pathlib import Path

import joblib
import pandas as pd
import torch

from ca_housing.model import HousingNet

RAW_COLUMNS = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
               "Population", "AveOccup", "Latitude", "Longitude"]


class Predictor:
    """Loads artifacts once, then serves predictions."""

    def __init__(self, models_dir):
        models_dir = Path(models_dir)

        self.preprocessor = joblib.load(models_dir / "preprocessor.joblib")

        with open(models_dir / "feature_names.json") as f:
            n_features = len(json.load(f))

        self.model = HousingNet(n_features=n_features)
        self.model.load_state_dict(torch.load(models_dir / "model.pt"))
        self.model.eval()

    def predict(self, rows):
        """rows: a dict, or a list of dicts. Returns list of floats."""
        if isinstance(rows, dict):
            rows = [rows]

        df = pd.DataFrame(rows)[RAW_COLUMNS]      # enforce column order
        X = self.preprocessor.transform(df).astype("float32")

        with torch.no_grad():
            preds = self.model(torch.from_numpy(X))

        return preds.tolist()