import json
from pathlib import Path

import torch

from ca_housing.model import HousingNet
from ca_housing.train import load_split, train, evaluate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

torch.manual_seed(42)

X_train, y_train = load_split(DATA / "train.npz")
X_val, y_val = load_split(DATA / "val.npz")

model = HousingNet(n_features=X_train.shape[1])

model, history = train(
    model,
    (X_train, y_train),
    (X_val, y_val),
    epochs=200,
    batch_size=64,
    lr=0.001,
    patience=15,
)

_, val_rmse = evaluate(model, X_val, y_val, torch.nn.MSELoss())
print(f"\nbest val RMSE: {val_rmse:.4f}  (${val_rmse*100_000:,.0f})")

torch.save(model.state_dict(), MODELS / "model.pt")
with open(MODELS / "history.json", "w") as f:
    json.dump(history, f, indent=2)

print("saved -> models/model.pt")


from ca_housing.metadata import write_metadata

meta = write_metadata(MODELS, model_version="0.1.0", extra={
    "n_features": int(X_train.shape[1]),
    "n_train_rows": int(len(X_train)),
    "epochs_run": len(history),
    "best_val_rmse": round(val_rmse, 4),
})
print("metadata:", meta)