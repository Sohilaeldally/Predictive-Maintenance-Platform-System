"""
Inference for the GRU model (used by the API when MODEL_TYPE=gru).

Loads models/gru_v1.pt (created by `python -m ml.gru_model`) and predicts the
RUL of one engine from its last WINDOW cycles. PyTorch is imported lazily, so
the default XGBoost setup does not need it installed.
"""
import numpy as np
import pandas as pd

from config import MODELS_DIR

GRU_MODEL_PATH = MODELS_DIR / "gru_v1.pt"


def build_regressor(n_features: int, hidden: int):
    import torch
    from torch import nn

    class GRURegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_features, hidden, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1]).squeeze(-1)

    return GRURegressor()


class GRUBundle:
    """The 3 trained GRUs plus the scaling info needed to use them."""

    def __init__(self, path=GRU_MODEL_PATH):
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.features = ckpt["features"]
        self.low = pd.Series(ckpt["low"])[self.features]
        self.span = pd.Series(ckpt["span"])[self.features]
        self.window = ckpt["window"]
        self.rul_cap = ckpt["rul_cap"]

        self.models = []
        for state in ckpt["states"]:
            model = build_regressor(len(self.features), ckpt["hidden"])
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)

    def predict_rul(self, df: pd.DataFrame) -> float:
        """df: readings of ONE engine (any length). Returns the predicted RUL."""
        import torch

        df = df.sort_values("time_cycles")
        scaled = ((df[self.features] - self.low) / self.span).to_numpy(dtype=np.float32)
        scaled = scaled[-self.window :]

        # Short history: repeat the first row (same padding used in training).
        missing = self.window - len(scaled)
        if missing > 0:
            scaled = np.vstack([np.repeat(scaled[:1], missing, axis=0), scaled])

        x = torch.from_numpy(scaled[None, :, :])
        with torch.no_grad():
            preds = [float(m(x)[0]) for m in self.models]

        return float(np.mean(preds) * self.rul_cap)
