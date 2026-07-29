from pathlib import Path
from ca_housing.predict import Predictor

ROOT = Path(__file__).resolve().parents[1]
predictor = Predictor(ROOT / "models")

house = {
    "MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.984127,
    "AveBedrms": 1.023810, "Population": 322.0, "AveOccup": 2.555556,
    "Latitude": 37.88, "Longitude": -122.23,
}

price = predictor.predict(house)[0]
print(f"predicted: {price:.3f}  (${price*100_000:,.0f})")
print(f"actual   : 4.526  ($452,600)")

# batch works too
cheap = dict(house, MedInc=1.5, Latitude=36.0, Longitude=-119.5)
for p in predictor.predict([house, cheap]):
    print(f"  {p:.3f}  ${p*100_000:,.0f}")