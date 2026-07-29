import json
from collections import deque
from pathlib import Path

import numpy as np

RAW_COLUMNS = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
               "Population", "AveOccup", "Latitude", "Longitude"]

# industry-standard interpretation
NO_DRIFT = 0.1
MODERATE = 0.25


def psi(ref_pct, bin_edges, incoming_values):
    """Population Stability Index of incoming values vs a saved reference."""
    edges = np.array(bin_edges, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf

    counts, _ = np.histogram(incoming_values, bins=edges)
    new_pct = counts / len(incoming_values)

    ref = np.clip(np.array(ref_pct), 1e-6, None)
    new = np.clip(new_pct, 1e-6, None)

    return float(np.sum((new - ref) * np.log(new / ref)))


def label(value):
    if value < NO_DRIFT:
        return "none"
    if value < MODERATE:
        return "moderate"
    return "significant"


class DriftMonitor:
    """Holds a rolling window of recent requests and scores it against training."""

    def __init__(self, reference_path, window=500, min_samples=50):
        with open(Path(reference_path)) as f:
            self.reference = json.load(f)["features"]

        self.window = {c: deque(maxlen=window) for c in RAW_COLUMNS}
        self.min_samples = min_samples
        self.window_size = window
        self.total_seen = 0

    def record(self, payload):
        for col in RAW_COLUMNS:
            self.window[col].append(float(payload[col]))
        self.total_seen += 1

    def report(self):
        n = len(self.window["MedInc"])

        if n < self.min_samples:
            return {
                "status": "insufficient_data",
                "samples_in_window": n,
                "samples_needed": self.min_samples,
                "total_seen": self.total_seen,
            }

        features = {}
        worst_value, worst_name = 0.0, None

        for col in RAW_COLUMNS:
            values = np.array(self.window[col])
            ref = self.reference[col]
            score = psi(ref["bin_pct"], ref["bin_edges"], values)

            features[col] = {
                "psi": round(score, 4),
                "drift": label(score),
                "window_mean": round(float(values.mean()), 3),
                "reference_mean": round(ref["mean"], 3),
            }

            if score > worst_value:
                worst_value, worst_name = score, col

        return {
            "status": "ok",
            "samples_in_window": n,
            "total_seen": self.total_seen,
            "overall_drift": label(worst_value),
            "worst_feature": worst_name,
            "worst_psi": round(worst_value, 4),
            "features": features,
        }