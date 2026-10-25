"""
Log every model version to MLflow so the experiments can be compared in a UI.

For each saved model (XGBoost v1, v2, v3 and the GRU) this evaluates it on the
FD001 test set (last cycle of each engine, true RUL from RUL_FD001.txt, the
same protocol as the README) and records:

  * params   : model family, features, RUL cap, hyper-parameters
  * metrics  : RMSE, MAE, R2, C-MAPSS score, MAE where true RUL <= 30
  * artifact : the model file

Run from the src/ folder (pip install mlflow):
    python -m ml.track_experiments
then browse the results from the project root with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
and open http://localhost:5000
"""
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import Booster, DMatrix

from config import BASE_DIR, DATA_DIR, MODELS_DIR, ROLLING_SENSORS, ROLLING_WINDOWS
from ml.gru_predict import GRUBundle
from ml.preprocessing import (
    clean_test_pipeline,
    clean_test_rolling_pipeline,
    clean_train_pipeline,
    clean_train_rolling_pipeline,
    get_feature_columns,
)
from ml.training import cmapss_score

EXPERIMENT = "PMPS-RUL-FD001"
ALERT_ZONE = 30

RAW = dict(
    train=str(DATA_DIR / "train_FD001.txt"),
    test=str(DATA_DIR / "test_FD001.txt"),
    rul=str(DATA_DIR / "RUL_FD001.txt"),
)

# Hyper-parameters copied from the training scripts (src/ml/xgboost_v*.py).
XGB_VERSIONS = {
    "xgboost_v1": dict(
        file="final_xgboost.json", rolling=False, rul_cap=None,
        params=dict(n_estimators=200, learning_rate=0.03, max_depth=4,
                    min_child_weight=3, subsample=1.0, colsample_bytree=0.8),
    ),
    "xgboost_v2": dict(
        file="final_xgboost_v2.json", rolling=True, rul_cap=None,
        params=dict(n_estimators=200, learning_rate=0.02, max_depth=3,
                    min_child_weight=3, subsample=0.8, colsample_bytree=0.8),
    ),
    "xgboost_v3": dict(
        file="final_xgboost_v3.json", rolling=True, rul_cap=125,
        params=dict(n_estimators=200, learning_rate=0.02, max_depth=3,
                    min_child_weight=3, subsample=0.8, colsample_bytree=0.8),
    ),
}


def last_cycle(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["unit_number", "time_cycles"])
        .groupby("unit_number").tail(1)
        .sort_values("unit_number")
    )


def metrics_for(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    zone = y_true <= ALERT_ZONE
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "cmapss_score": float(cmapss_score(y_true, y_pred)),
        f"mae_rul_le_{ALERT_ZONE}": float(np.abs(y_true[zone] - y_pred[zone]).mean()),
    }


def log_run(name, params, metrics, artifact_path, tags=None):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"model": name, **(tags or {})})
        mlflow.log_artifact(str(artifact_path), artifact_path="model")
    print(f"{name:11s} " + "  ".join(f"{k}={v:.2f}" for k, v in metrics.items()))


def main():
    mlflow.set_tracking_uri("sqlite:///" + (BASE_DIR / "mlflow.db").as_posix())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=(BASE_DIR / "mlartifacts").as_uri()
        )
    mlflow.set_experiment(EXPERIMENT)

    test_plain = last_cycle(clean_test_pipeline(RAW["test"], RAW["rul"]))
    test_roll_all = clean_test_rolling_pipeline(
        RAW["test"], RAW["rul"], sensors=ROLLING_SENSORS, windows=ROLLING_WINDOWS
    )
    test_roll = last_cycle(test_roll_all)
    # Feature lists exactly as used at training time.
    cols_plain = get_feature_columns(clean_train_pipeline(RAW["train"]))
    cols_roll = get_feature_columns(
        clean_train_rolling_pipeline(RAW["train"], sensors=ROLLING_SENSORS, windows=ROLLING_WINDOWS)
    )

    for name, cfg in XGB_VERSIONS.items():
        data, cols = (test_roll, cols_roll) if cfg["rolling"] else (test_plain, cols_plain)
        path = MODELS_DIR / cfg["file"]
        booster = Booster()
        booster.load_model(path)
        pred = booster.predict(DMatrix(data[cols]))
        params = {
            "family": "xgboost",
            "n_features": len(cols),
            "rolling_features": cfg["rolling"],
            "rul_cap": cfg["rul_cap"] if cfg["rul_cap"] else "none",
            **cfg["params"],
        }
        log_run(name, params, metrics_for(data["RUL"], pred), path)

    # GRU: evaluate the saved checkpoint (no retraining).
    gru_path = MODELS_DIR / "gru_v1.pt"
    gru = GRUBundle(gru_path)
    y_true, y_pred = [], []
    for _, g in test_roll_all.groupby("unit_number"):
        g = g.sort_values("time_cycles")
        y_true.append(g["RUL"].iloc[-1])
        y_pred.append(gru.predict_rul(g))
    params = {
        "family": "gru",
        "n_features": len(gru.features),
        "rolling_features": False,
        "rul_cap": gru.rul_cap,
        "window": gru.window,
        "hidden": 64,
        "layers": 2,
        "seeds_averaged": len(gru.models),
    }
    log_run("gru_v1", params, metrics_for(y_true, y_pred), gru_path)
    print("\nDone. Browse with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()