"""
Explain the XGBoost v3 model with SHAP.

SHAP gives every feature a contribution to each prediction: how many cycles it
pushed the predicted RUL up (+) or down (-) compared with the average
prediction. Two views are saved to docs/images/:

  shap_summary.png   global view: which features matter most, and in which
                     direction (high sensor value -> lower or higher RUL)
  shap_engine34.png  local view: why engine 34 is predicted close to failure

Run from the src/ folder (pip install shap matplotlib):
    python -m ml.explain
"""
import matplotlib

matplotlib.use("Agg")  # save to files, no window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import Booster

from config import (
    BASE_DIR,
    DATA_DIR,
    MODEL_FEATURES,
    MODELS_DIR,
    ROLLING_SENSORS,
    ROLLING_WINDOWS,
)
from ml.preprocessing import clean_test_rolling_pipeline, clean_train_rolling_pipeline

IMAGES_DIR = BASE_DIR / "docs" / "images"
SAMPLE_SIZE = 3000
EXPLAINED_ENGINE = 34

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

booster = Booster()
booster.load_model(MODELS_DIR / "final_xgboost_v3.json")
explainer = shap.TreeExplainer(booster)

# ---- Global view: a random sample of training rows -----------------------
train = clean_train_rolling_pipeline(
    str(DATA_DIR / "train_FD001.txt"), sensors=ROLLING_SENSORS, windows=ROLLING_WINDOWS
)
sample = train[MODEL_FEATURES].sample(SAMPLE_SIZE, random_state=42)
shap_values = explainer.shap_values(sample)

importance = (
    pd.Series(np.abs(shap_values).mean(axis=0), index=MODEL_FEATURES)
    .sort_values(ascending=False)
)
print("Top 10 features by mean |SHAP| (average cycles of impact on the prediction):")
print(importance.head(10).round(2).to_string())

plt.figure()
shap.summary_plot(shap_values, sample, max_display=15, show=False)
plt.title("What drives the predicted RUL (XGBoost v3)")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- Local view: last observed cycle of one test engine -------------------
test = clean_test_rolling_pipeline(
    str(DATA_DIR / "test_FD001.txt"),
    str(DATA_DIR / "RUL_FD001.txt"),
    sensors=ROLLING_SENSORS,
    windows=ROLLING_WINDOWS,
)
row = (
    test[test["unit_number"] == EXPLAINED_ENGINE]
    .sort_values("time_cycles")
    .tail(1)[MODEL_FEATURES]
)
explanation = explainer(row)
predicted = float(booster.predict(__import__("xgboost").DMatrix(row))[0])
print(f"\nEngine {EXPLAINED_ENGINE}: predicted RUL {predicted:.1f} "
      f"(average prediction {float(np.ravel(explanation.base_values)[0]):.1f})")

plt.figure()
shap.plots.waterfall(explanation[0], max_display=10, show=False)
plt.title(f"Why engine {EXPLAINED_ENGINE} is predicted close to failure")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "shap_engine34.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved images to {IMAGES_DIR}")