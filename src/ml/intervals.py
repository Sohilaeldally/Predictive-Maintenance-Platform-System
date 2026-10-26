"""
Prediction intervals: "RUL = 20, probably between 14 and 27", not just "20".

Method: split-conformal prediction, done separately for groups ("bins") of
the predicted RUL. In short:

  1. Take predictions the model made on engines it did NOT train on, and look
     at how wrong it was (|predicted - true|).
  2. For each bin of predicted RUL, keep the error that 80% of those
     predictions stayed within (the "half width").
  3. At prediction time the interval is  predicted +- half width of its bin.

Bins matter because the model is far more accurate near failure than far from
it, so an engine predicted at 10 cycles gets a much narrower interval than one
predicted at 100.

Where the "unseen" predictions come from:
  * XGBoost : out-of-fold predictions of the 5-fold GroupKFold (every training
              engine is predicted by a model that never saw it).
  * GRU     : the 20 validation engines (not used to fit the weights; they
              only picked the best epoch).

The result is written to models/intervals_<model>.json and read by the API
(no retraining, no database change). Coverage is then CHECKED on the test set
(100 engines, last cycle, true RUL from RUL_FD001.txt), which was not used to
build the intervals.

Run from the src/ folder:
    python -m ml.intervals            # builds both (GRU needs PyTorch)
    python -m ml.intervals gru        # or only one of: gru | xgboost
"""
import json
import math
import sys

import numpy as np
import pandas as pd

from config import DATA_DIR, MODELS_DIR, ROLLING_SENSORS, ROLLING_WINDOWS

COVERAGE = 0.80          # target: the true RUL falls inside the interval 80% of the time
RUL_CAP = 125
BIN_EDGES = [0, 15, 30, 60, 100, float("inf")]   # bins of the PREDICTED RUL
MIN_PER_BIN = 30         # a bin with fewer points is too noisy to trust

RAW_TRAIN = str(DATA_DIR / "train_FD001.txt")
RAW_TEST = str(DATA_DIR / "test_FD001.txt")
RAW_RUL = str(DATA_DIR / "RUL_FD001.txt")


def interval_path(model_type: str):
    return MODELS_DIR / f"intervals_{model_type}.json"


# ---------------------------------------------------------------- build ----
def half_widths(pred, true) -> tuple[list[float], list[int]]:
    """Per bin of predicted RUL: the error 80% of predictions stayed within."""
    pred, err = np.asarray(pred, dtype=float), np.abs(np.asarray(true) - np.asarray(pred))
    widths, counts = [], []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        e = err[(pred >= lo) & (pred < hi)]
        counts.append(int(len(e)))
        if len(e) < MIN_PER_BIN:
            widths.append(None)          # filled below from a neighbouring bin
            continue
        # conformal quantile: slightly above 80% so the guarantee holds for small n
        level = min(1.0, math.ceil((len(e) + 1) * COVERAGE) / len(e))
        widths.append(float(np.quantile(e, level)))
    # a bin with too little data borrows from the next wider bin that has data
    for i in range(len(widths) - 1, -1, -1):
        if widths[i] is None:
            widths[i] = next((w for w in widths[i + 1:] if w is not None), None)
            if widths[i] is None:
                widths[i] = next(w for w in widths if w is not None)
    return widths, counts


def save_intervals(model_type: str, pred, true, source: str) -> dict:
    widths, counts = half_widths(pred, true)
    payload = {
        "model": model_type,
        "coverage": COVERAGE,
        "bin_edges": [e if e != float("inf") else 1e9 for e in BIN_EDGES],
        "half_width": [round(w, 2) for w in widths],
        "n_per_bin": counts,
        "calibration_source": source,
        "rul_cap": RUL_CAP,
    }
    MODELS_DIR.mkdir(exist_ok=True)
    interval_path(model_type).write_text(json.dumps(payload, indent=2))
    return payload


def calibration_xgboost():
    """Out-of-fold (predicted, true capped RUL) for every training engine."""
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBRegressor

    from ml.cross_validation import MODEL_PARAMS, N_SPLITS
    from ml.preprocessing import cap_rul, clean_train_rolling_pipeline, get_feature_columns

    train = cap_rul(
        clean_train_rolling_pipeline(RAW_TRAIN, sensors=ROLLING_SENSORS, windows=ROLLING_WINDOWS),
        cap=RUL_CAP,
    )
    cols = get_feature_columns(train)
    X, y, groups = train[cols], train["RUL"].to_numpy(), train["unit_number"]
    oof = np.zeros(len(train))
    for tr, va in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        model = XGBRegressor(**MODEL_PARAMS).fit(X.iloc[tr], y[tr])
        oof[va] = model.predict(X.iloc[va])
    return oof, y, f"{N_SPLITS}-fold GroupKFold out-of-fold, {train['unit_number'].nunique()} engines"


def calibration_gru(n_folds: int = 5):
    """Out-of-fold GRU predictions for every training engine (needs PyTorch).

    For each fold a fresh GRU (one seed) is trained on the other engines; 10%
    of those engines are held back only to pick the best epoch, so the fold's
    own engines are never touched until they are predicted. Takes ~10 minutes
    on CPU. The deployed model averages 3 seeds, which is a little more
    accurate than one seed, so the intervals are slightly conservative.
    """
    from sklearn.model_selection import GroupKFold, train_test_split

    from ml.gru_model import make_windows, predict, train_one
    from ml.gru_predict import GRUBundle
    from ml.preprocessing import add_rul_train, cap_rul, load_raw_data

    bundle = GRUBundle()
    train = cap_rul(add_rul_train(load_raw_data(RAW_TRAIN)), RUL_CAP)
    train[bundle.features] = (train[bundle.features] - bundle.low) / bundle.span

    engines = train["unit_number"].to_numpy()
    preds, trues = [], []
    for fold, (tr_idx, oof_idx) in enumerate(
        GroupKFold(n_splits=n_folds).split(train, groups=engines), start=1
    ):
        tr_engines = np.unique(engines[tr_idx])
        fit_engines, stop_engines = train_test_split(tr_engines, test_size=0.1, random_state=fold)
        fit_df = train[train["unit_number"].isin(fit_engines)]
        stop_df = train[train["unit_number"].isin(stop_engines)]
        oof_df = train.iloc[oof_idx]

        print(f"fold {fold}/{n_folds}: train {len(fit_engines)} engines, "
              f"early-stop {len(stop_engines)}, predict {oof_df['unit_number'].nunique()}")
        X_tr, y_tr, _ = make_windows(fit_df)
        X_st, y_st, _ = make_windows(stop_df)
        model = train_one(X_tr, y_tr, X_st, y_st, seed=fold)

        X_oof, y_oof, _ = make_windows(oof_df)
        preds.append(predict(model, X_oof))
        trues.append(y_oof)

    return (
        np.concatenate(preds),
        np.concatenate(trues),
        f"{n_folds}-fold GroupKFold out-of-fold GRU, {train['unit_number'].nunique()} engines",
    )


# ------------------------------------------------------------------ use ----
_cache: dict = {}


def load_intervals(model_type: str):
    """The saved interval table, or None if it has not been built yet."""
    if model_type not in _cache:
        path = interval_path(model_type)
        _cache[model_type] = json.loads(path.read_text()) if path.exists() else None
    return _cache[model_type]


def add_interval(predicted_rul: float, model_type: str) -> dict:
    """{'rul_lower', 'rul_upper', 'interval_coverage'} (values None if no table)."""
    table = load_intervals(model_type)
    if table is None:
        return {"rul_lower": None, "rul_upper": None, "interval_coverage": None}
    edges, widths = table["bin_edges"], table["half_width"]
    i = max(0, min(np.searchsorted(edges, predicted_rul, side="right") - 1, len(widths) - 1))
    w = widths[i]
    return {
        "rul_lower": round(max(0.0, predicted_rul - w), 1),
        "rul_upper": round(predicted_rul + w, 1),
        "interval_coverage": table["coverage"],
    }


# ---------------------------------------------------------------- check ----
def test_set_predictions(model_type: str):
    """(predicted, true RUL) at the last cycle of each of the 100 test engines."""
    from ml.preprocessing import clean_test_rolling_pipeline

    test_all = clean_test_rolling_pipeline(
        RAW_TEST, RAW_RUL, sensors=ROLLING_SENSORS, windows=ROLLING_WINDOWS
    )
    if model_type == "gru":
        from ml.gru_predict import GRUBundle

        bundle = GRUBundle()
        rows = [
            (bundle.predict_rul(g), g.sort_values("time_cycles")["RUL"].iloc[-1])
            for _, g in test_all.groupby("unit_number")
        ]
        pred, true = zip(*rows)
        return np.array(pred), np.array(true)

    from xgboost import Booster, DMatrix

    from config import MODEL_FEATURES

    last = test_all.sort_values(["unit_number", "time_cycles"]).groupby("unit_number").tail(1)
    booster = Booster()
    booster.load_model(MODELS_DIR / "final_xgboost_v3.json")
    return booster.predict(DMatrix(last[MODEL_FEATURES])), last["RUL"].to_numpy()


def report_coverage(model_type: str) -> dict:
    _cache.pop(model_type, None)
    pred, true = test_set_predictions(model_type)
    lo_hi = [add_interval(p, model_type) for p in pred]
    lower = np.array([d["rul_lower"] for d in lo_hi])
    upper = np.array([d["rul_upper"] for d in lo_hi])
    inside = (true >= lower) & (true <= upper)
    zone = true <= 30
    return {
        "coverage_all": float(inside.mean()),
        "coverage_alert_zone": float(inside[zone].mean()),
        "n_alert_zone": int(zone.sum()),
        "mean_width": float((upper - lower).mean()),
        "mean_width_alert_zone": float((upper - lower)[zone].mean()),
    }


def build(model_type: str) -> None:
    pred, true, source = calibration_gru() if model_type == "gru" else calibration_xgboost()
    table = save_intervals(model_type, pred, true, source)
    print(f"\n=== {model_type}: {int(COVERAGE * 100)}% intervals, built from {source} ===")
    print(f"{'predicted RUL':>14} {'n':>7} {'+- cycles':>10}")
    edges = table["bin_edges"]
    for i, (w, n) in enumerate(zip(table["half_width"], table["n_per_bin"])):
        hi = "inf" if edges[i + 1] >= 1e9 else int(edges[i + 1])
        print(f"{f'{int(edges[i])}-{hi}':>14} {n:>7} {w:>10.1f}")

    result = report_coverage(model_type)
    print(f"\nCheck on the TEST set (100 engines, last cycle, never used above):")
    print(f"  coverage, all engines          : {result['coverage_all']:.0%}  (target {COVERAGE:.0%})")
    print(
        f"  coverage, true RUL <= 30 "
        f"({result['n_alert_zone']} engines): {result['coverage_alert_zone']:.0%}"
    )
    print(f"  mean interval width            : {result['mean_width']:.1f} cycles")
    print(f"  mean width, true RUL <= 30     : {result['mean_width_alert_zone']:.1f} cycles")


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["xgboost", "gru"]
    for name in wanted:
        build(name)
