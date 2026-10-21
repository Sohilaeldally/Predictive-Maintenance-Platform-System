"""
Engine-level cross-validation for the v3 model.

Rows of the same engine are highly correlated, so a normal random K-fold would
leak information: the model would see cycles 1-100 of an engine in training and
be tested on cycle 101 of the same engine. GroupKFold keeps every engine
entirely inside ONE fold, so each fold is tested on engines the model has
never seen.

Run from the src/ folder:
    python -m ml.cross_validation
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from config import DATA_DIR, ROLLING_SENSORS, ROLLING_WINDOWS
from ml.preprocessing import (
    cap_rul,
    clean_train_rolling_pipeline,
    get_feature_columns,
)
from ml.training import evaluate_model

N_SPLITS = 5
RUL_CAP = 125
ALERT_ZONE = 30  # also report the error where it matters for alerts

MODEL_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.02,
    max_depth=3,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)


def run_cross_validation() -> pd.DataFrame:
    train = clean_train_rolling_pipeline(
        str(DATA_DIR / "train_FD001.txt"),
        sensors=ROLLING_SENSORS,
        windows=ROLLING_WINDOWS,
    )
    train = cap_rul(train, cap=RUL_CAP)

    feature_cols = get_feature_columns(train)
    X = train[feature_cols]
    y = train["RUL"]
    groups = train["unit_number"]

    results = []
    splitter = GroupKFold(n_splits=N_SPLITS)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        # Safety check: no engine may appear on both sides.
        assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[val_idx])

        model = XGBRegressor(**MODEL_PARAMS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[val_idx])

        metrics = evaluate_model(y.iloc[val_idx], pred, model_name=f"Fold {fold}")
        metrics["val_engines"] = groups.iloc[val_idx].nunique()

        y_true = y.iloc[val_idx].values
        zone = y_true <= ALERT_ZONE
        metrics["MAE_alert_zone"] = float(np.abs(y_true[zone] - pred[zone]).mean())
        results.append(metrics)

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_cross_validation()

    print("\n=== Engine-level GroupKFold results ===")
    print(df[["model", "val_engines", "RMSE", "MAE", "R2", "MAE_alert_zone"]].round(2).to_string(index=False))

    print("\nMean +- std over folds")
    for col in ["RMSE", "MAE", "R2", "MAE_alert_zone"]:
        print(f"{col:15s}: {df[col].mean():.2f} +- {df[col].std():.2f}")