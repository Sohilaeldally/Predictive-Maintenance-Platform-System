"""
GRU baseline for RUL prediction (comparison against the XGBoost v3 model).

XGBoost sees one row at a time (plus hand-made rolling features). A GRU reads
the last WINDOW cycles of raw sensors as a sequence and learns the temporal
pattern itself, with no manual rolling features.

Same protocol as the XGBoost evaluation, so the numbers are comparable:
  * training RUL capped at 125
  * scored on the TEST set at the last observed cycle of each engine
    (true RUL taken from RUL_FD001.txt, not capped)

Run from the src/ folder (needs PyTorch: pip install torch):
    python -m ml.gru_model
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from config import DATA_DIR, MODEL_FEATURES
from ml.preprocessing import add_rul_test, add_rul_train, cap_rul, load_raw_data
from ml.training import cmapss_score, evaluate_model, split_by_engine

WINDOW = 30
RUL_CAP = 125
HIDDEN = 64
EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
N_SEEDS = 3

# Raw inputs only: no rolling features (the GRU learns those itself) and no
# time_cycles (keep the comparison about sensor behaviour).
FEATURES = [c for c in MODEL_FEATURES if "roll" not in c and c != "time_cycles"]


def make_windows(df: pd.DataFrame, last_only: bool = False):
    """Return (X, y, engine_ids): one window of WINDOW cycles per row.

    Early cycles are padded by repeating the engine's first row, so every
    cycle (and every short test engine) gets a full-length window.
    """
    X, y, ids = [], [], []
    for unit, g in df.sort_values(["unit_number", "time_cycles"]).groupby("unit_number"):
        values = g[FEATURES].to_numpy(dtype=np.float32)
        padded = np.vstack([np.repeat(values[:1], WINDOW - 1, axis=0), values])
        targets = g["RUL"].to_numpy(dtype=np.float32)
        positions = [len(values) - 1] if last_only else range(len(values))
        for i in positions:
            X.append(padded[i : i + WINDOW])
            y.append(targets[i])
            ids.append(unit)
    return np.stack(X), np.array(y, dtype=np.float32), np.array(ids)


class GRURegressor(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.gru = nn.GRU(n_features, HIDDEN, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)  # use the last time step


def predict(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy() * RUL_CAP


def train_one(X_tr, y_tr, X_va, y_va, seed: int) -> GRURegressor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GRURegressor(X_tr.shape[2])
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    X_tr_t, y_tr_t = torch.from_numpy(X_tr), torch.from_numpy(y_tr / RUL_CAP)
    best_state, best_val = None, float("inf")

    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(X_tr_t))
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            optimiser.zero_grad()
            loss = loss_fn(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            optimiser.step()

        val_rmse = float(np.sqrt(np.mean((predict(model, X_va) - y_va) ** 2)))
        if val_rmse < best_val:  # keep the best epoch (early stopping)
            best_val = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"  seed {seed}: best validation RMSE = {best_val:.2f}")
    return model


if __name__ == "__main__":
    train = cap_rul(add_rul_train(load_raw_data(str(DATA_DIR / "train_FD001.txt"))), RUL_CAP)
    test = add_rul_test(
        load_raw_data(str(DATA_DIR / "test_FD001.txt")), str(DATA_DIR / "RUL_FD001.txt")
    )

    # Scale with training statistics only.
    low, high = train[FEATURES].min(), train[FEATURES].max()
    span = (high - low).replace(0, 1)
    train[FEATURES] = (train[FEATURES] - low) / span
    test[FEATURES] = (test[FEATURES] - low) / span

    # Validation engines are only used to pick the best epoch.
    tr_df, va_df = split_by_engine(train, test_size=0.2, random_state=42)
    X_tr, y_tr, _ = make_windows(tr_df)
    X_va, y_va, _ = make_windows(va_df)
    X_te, y_te, _ = make_windows(test, last_only=True)
    print(f"train windows {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")

    preds = []
    for seed in range(N_SEEDS):
        model = train_one(X_tr, y_tr, X_va, y_va, seed)
        preds.append(predict(model, X_te))

    # Average of several seeds: smoother and more reliable than one run.
    y_pred = np.mean(preds, axis=0)

    print("\n=== GRU on the TEST set (last cycle of each engine) ===")
    evaluate_model(y_te, y_pred, "GRU (mean of seeds)")
    print("C-MAPSS score:", round(float(cmapss_score(y_te, y_pred)), 1))