import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def load_split(path):
    d = np.load(path)
    return torch.from_numpy(d["X"]), torch.from_numpy(d["y"])


def evaluate(model, X, y, loss_fn):
    """Loss + RMSE on a whole split, no gradient tracking."""
    model.eval()
    with torch.no_grad():
        preds = model(X)
        loss = loss_fn(preds, y).item()
        rmse = torch.sqrt(((preds - y) ** 2).mean()).item()
    return loss, rmse


def train(model, train_data, val_data, epochs=100, batch_size=64,
          lr=0.001, patience=10, verbose=True):

    X_train, y_train = train_data
    X_val, y_val = val_data

    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=batch_size, shuffle=True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_weights = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):

        model.train()
        running = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(X_batch)

        train_loss = running / len(X_train)
        val_loss, val_rmse = evaluate(model, X_val, y_val, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_rmse": val_rmse})

        if val_loss < best_val:
            best_val = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            marker = " *"
        else:
            bad_epochs += 1
            marker = ""

        if verbose and (epoch % 5 == 0 or marker):
            print(f"epoch {epoch:3d}  train {train_loss:.4f}  "
                  f"val {val_loss:.4f}  rmse {val_rmse:.4f}{marker}")

        if bad_epochs >= patience:
            print(f"\nearly stop at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_weights)
    return model, history