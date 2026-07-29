import json
from pathlib import Path

import numpy as np
import pytest

from ca_housing.drift import DriftMonitor, psi

MODELS = Path(__file__).resolve().parents[1] / "models"
REF = MODELS / "reference_stats.json"

pytestmark = pytest.mark.skipif(
    not REF.exists(), reason="run scripts/reference_stats.py first")

BASE = {"MedInc": 4.0, "HouseAge": 30.0, "AveRooms": 5.0, "AveBedrms": 1.0,
        "Population": 1200.0, "AveOccup": 3.0, "Latitude": 35.0,
        "Longitude": -119.0}

DRIFTED = {"MedInc": 13.0, "HouseAge": 5.0, "AveRooms": 9.0, "AveBedrms": 1.0,
           "Population": 500.0, "AveOccup": 2.0, "Latitude": 34.0,
           "Longitude": -118.0}


def load_reference(feature="MedInc"):
    with open(REF) as f:
        return json.load(f)["features"][feature]


def test_psi_low_for_same_distribution():
    ref = load_reference()
    edges = np.array(ref["bin_edges"])
    mid = (edges[:-1] + edges[1:]) / 2
    values = np.repeat(mid, 200)
    assert psi(ref["bin_pct"], ref["bin_edges"], values) < 0.1


def test_psi_high_for_shifted_distribution():
    ref = load_reference()
    shifted = np.full(500, 12.0)
    assert psi(ref["bin_pct"], ref["bin_edges"], shifted) > 1.0


def test_monitor_refuses_small_samples():
    m = DriftMonitor(REF, window=500, min_samples=200)
    for _ in range(10):
        m.record(BASE)
    assert m.report()["status"] == "insufficient_data"


def test_monitor_detects_drift():
    m = DriftMonitor(REF, window=500, min_samples=200)
    for _ in range(300):
        m.record(DRIFTED)
    report = m.report()
    assert report["overall_drift"] == "significant"
    assert report["worst_psi"] > 1.0


def test_window_is_bounded():
    m = DriftMonitor(REF, window=50, min_samples=10)
    for _ in range(200):
        m.record(BASE)
    report = m.report()
    assert report["samples_in_window"] == 50
    assert report["total_seen"] == 200