import json
from pathlib import Path

import numpy as np
import torch

from ca_housing.model import HousingNet
from ca_housing.train import load_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

X_train, y_train = load_split(DATA / "train.npz")
X_test, y_test = load_split(DATA / "test.npz")

# rebuild the architecture, then load the learned numbers into it
model = HousingNet(n_features=X_train.shape[1])
model.load_state_dict(torch.load(MODELS / "model.pt"))
model.eval()

with torch.no_grad():
    preds = model(X_test)

err = preds - y_test
rmse = torch.sqrt((err ** 2).mean()).item()
mae = err.abs().mean().item()

# how much of the target's variation the model explains, 0..1
ss_res = (err ** 2).sum().item()
ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
r2 = 1 - ss_res / ss_tot

# baseline: always predict the training mean
baseline_rmse = torch.sqrt(((y_test - y_train.mean()) ** 2).mean()).item()

print(f"{'':12s}{'RMSE':>8s}{'in $':>12s}")
print(f"{'baseline':12s}{baseline_rmse:8.4f}{baseline_rmse*100_000:12,.0f}")
print(f"{'model':12s}{rmse:8.4f}{rmse*100_000:12,.0f}")
print(f"\nMAE : {mae:.4f}  (${mae*100_000:,.0f})")
print(f"R^2 : {r2:.4f}")

# where does it do badly?
capped = y_test >= 5.0
print(f"\nRMSE on normal rows : {torch.sqrt((err[~capped]**2).mean()):.4f}")
print(f"RMSE on capped rows : {torch.sqrt((err[capped]**2).mean()):.4f}"
      f"  ({capped.sum().item()} rows)")

with open(MODELS / "metrics.json", "w") as f:
    json.dump({"test_rmse": rmse, "test_mae": mae, "test_r2": r2}, f, indent=2)
print("\nsaved -> models/metrics.json")