from pathlib import Path

import pytest

from ca_housing.predict import Predictor

MODELS = Path(__file__).resolve().parents[1] / "models"

HOUSE = {
    "MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.984127,
    "AveBedrms": 1.023810, "Population": 322.0, "AveOccup": 2.555556,
    "Latitude": 37.88, "Longitude": -122.23,
}


@pytest.fixture(scope="module")
def predictor():
    return Predictor(MODELS)


def test_returns_sensible_value(predictor):
    price = predictor.predict(HOUSE)[0]
    assert 0.1 < price < 6.0


def test_batch_matches_single(predictor):
    single = predictor.predict(HOUSE)[0]
    batch = predictor.predict([HOUSE, HOUSE])
    assert batch[0] == pytest.approx(single)
    assert len(batch) == 2


def test_column_order_does_not_matter(predictor):
    shuffled = dict(reversed(list(HOUSE.items())))
    assert predictor.predict(shuffled)[0] == pytest.approx(
        predictor.predict(HOUSE)[0]
    )


def test_income_increases_price(predictor):
    poor = predictor.predict(dict(HOUSE, MedInc=1.5))[0]
    rich = predictor.predict(dict(HOUSE, MedInc=12.0))[0]
    assert rich > poor