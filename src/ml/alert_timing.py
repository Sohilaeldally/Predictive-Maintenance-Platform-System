"""
When does each model raise the alert, compared with when it should have?

For every test engine we replay its life cycle by cycle (using only the data
available up to each cycle) and find the first cycle where the model predicts
RUL <= threshold. The "right" cycle is failure_cycle - threshold, where
failure_cycle = last observed cycle + true RUL (from RUL_FD001.txt).

    lag = first alert cycle - right cycle
    lag > 0  -> the alert came LATE   (dangerous: less time to react)
    lag < 0  -> the alert came EARLY  (safe, but may waste maintenance)

Run from src/ (needs models/gru_v1.pt):
    python -m ml.alert_timing
"""
import numpy as np
import pandas as pd
import torch
from xgboost import Booster, DMatrix

from config import DATA_DIR, MODEL_FEATURES, MODELS_DIR, ROLLING_SENSORS, ROLLING_WINDOWS
from ml.gru_predict import GRUBundle
from ml.preprocessing import add_rolling_features, load_raw_data

THRESHOLDS = {"CRITICAL (<=15)": 15, "WARNING (<=30)": 30}
ON_TIME = 5  # cycles of tolerance either way

test = load_raw_data(str(DATA_DIR / "test_FD001.txt")).sort_values(["unit_number", "time_cycles"])
true_rul = pd.read_csv(DATA_DIR / "RUL_FD001.txt", header=None)[0].to_numpy()

booster = Booster()
booster.load_model(MODELS_DIR / "final_xgboost_v3.json")
gru = GRUBundle()

# Per-cycle predictions of both models for every engine.
xgb_rows = add_rolling_features(test, ROLLING_SENSORS, ROLLING_WINDOWS).sort_values(
    ["unit_number", "time_cycles"]
)
xgb_rows["pred"] = booster.predict(DMatrix(xgb_rows[MODEL_FEATURES]))


def gru_predictions(g: pd.DataFrame) -> np.ndarray:
    scaled = ((g[gru.features] - gru.low) / gru.span).to_numpy(dtype=np.float32)
    padded = np.vstack([np.repeat(scaled[:1], gru.window - 1, axis=0), scaled])
    windows = np.stack([padded[i : i + gru.window] for i in range(len(scaled))])
    with torch.no_grad():
        outs = [m(torch.from_numpy(windows)).numpy() for m in gru.models]
    return np.mean(outs, axis=0) * gru.rul_cap


results = []
for unit, g in test.groupby("unit_number"):
    cycles = g["time_cycles"].to_numpy()
    failure_cycle = cycles[-1] + true_rul[unit - 1]
    preds = {
        "xgboost": xgb_rows.loc[xgb_rows["unit_number"] == unit, "pred"].to_numpy(),
        "gru": gru_predictions(g),
    }
    for model, p in preds.items():
        for label, t in THRESHOLDS.items():
            right = failure_cycle - t
            hit = np.where(p <= t)[0]
            first = cycles[hit[0]] if len(hit) else None
            results.append(
                {
                    "model": model,
                    "level": label,
                    "engine": unit,
                    "crossed": right <= cycles[-1],  # true crossing is inside the data
                    "lag": None if first is None else int(first - right),
                }
            )

df = pd.DataFrame(results)

for label in THRESHOLDS:
    print(f"\n===== {label} =====")
    for model in ["xgboost", "gru"]:
        d = df[(df.model == model) & (df.level == label)]
        real = d[d.crossed]                       # engines really at this level
        flagged = real.dropna(subset=["lag"])
        missed = len(real) - len(flagged)
        late = int((flagged.lag > ON_TIME).sum())
        ok = int((flagged.lag.abs() <= ON_TIME).sum())
        early = int((flagged.lag < -ON_TIME).sum())
        early_false = d[~d.crossed].dropna(subset=["lag"])  # flagged before reaching it
        print(
            f"{model:8s} engines really there: {len(real):2d} | "
            f"on time (+-{ON_TIME}): {ok:2d} | late (>{ON_TIME}): {late:2d} | "
            f"early (>{ON_TIME}): {early:2d} | never flagged: {missed:2d} | "
            f"median lag: {flagged.lag.median():+.1f} cycles"
        )
        print(
            f"{'':8s} engines NOT yet there but flagged: {len(early_false):2d} "
            f"(median {early_false.lag.median():+.1f} cycles early)"
            if len(early_false)
            else f"{'':8s} engines NOT yet there but flagged:  0"
        )