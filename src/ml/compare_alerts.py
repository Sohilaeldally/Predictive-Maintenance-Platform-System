"""
Compare XGBoost v3 and the GRU on what matters for alerts.

For engines that are really close to failure (true RUL <= CRITICAL) how many
does each model flag, and how many healthy engines does it flag by mistake?
Scored on the test set at each engine's last cycle.

Run from src/ (needs models/gru_v1.pt from `python -m ml.gru_model`):
    python -m ml.compare_alerts
"""
import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix

from config import DATA_DIR, MODEL_FEATURES, MODELS_DIR, ROLLING_SENSORS, ROLLING_WINDOWS
from ml.gru_predict import GRUBundle
from ml.preprocessing import add_rolling_features, add_rul_test, load_raw_data

CRITICAL = 15
WARNING = 30

test = add_rul_test(
    load_raw_data(str(DATA_DIR / "test_FD001.txt")), str(DATA_DIR / "RUL_FD001.txt")
)
with_roll = add_rolling_features(test, ROLLING_SENSORS, ROLLING_WINDOWS)

booster = Booster()
booster.load_model(MODELS_DIR / "final_xgboost_v3.json")
gru = GRUBundle()

rows = []
for unit, g in test.groupby("unit_number"):
    last_roll = with_roll[with_roll["unit_number"] == unit].sort_values("time_cycles").tail(1)
    rows.append(
        {
            "engine": unit,
            "true": float(g.sort_values("time_cycles")["RUL"].iloc[-1]),
            "xgboost": float(booster.predict(DMatrix(last_roll[MODEL_FEATURES]))[0]),
            "gru": gru.predict_rul(g),
        }
    )
df = pd.DataFrame(rows)


def report(threshold: int, label: str):
    truly = df["true"] <= threshold
    print(f"\n{label} (RUL <= {threshold}): {int(truly.sum())} engines are truly at this level")
    for model in ["xgboost", "gru"]:
        flagged = df[model] <= threshold
        caught = int((flagged & truly).sum())
        false = int((flagged & ~truly).sum())
        print(f"  {model:8s} caught {caught}/{int(truly.sum())}   false alarms: {false}")


report(CRITICAL, "CRITICAL")
report(WARNING, "WARNING")

alert_zone = df["true"] <= WARNING
print("\nMAE where true RUL <= 30 (alert zone):")
for model in ["xgboost", "gru"]:
    print(f"  {model:8s} {np.abs(df.loc[alert_zone, model] - df.loc[alert_zone, 'true']).mean():.2f}")
