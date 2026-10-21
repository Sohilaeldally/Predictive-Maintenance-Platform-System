import os

import pandas as pd
from xgboost import Booster, DMatrix

from ml.preprocessing import add_rolling_features
from config import ROLLING_SENSORS, ROLLING_WINDOWS, MODEL_FEATURES, MODELS_DIR

from db.queries import get_engine_history,save_prediction

MODEL_PATH = MODELS_DIR / "final_xgboost_v3.json"

# Which model the API serves: "xgboost" (default) or "gru" (needs PyTorch).
MODEL_TYPE = os.environ.get("MODEL_TYPE", "xgboost").lower()


def load_model():
    if MODEL_TYPE == "gru":
        from ml.gru_predict import GRUBundle

        return GRUBundle()

    model = Booster()
    model.load_model(MODEL_PATH)
    return model


def predict_rul(df: pd.DataFrame, model) -> float:
    if MODEL_TYPE == "gru":
        return model.predict_rul(df)

    df = df.copy()

    df = df.sort_values(
        ["unit_number", "time_cycles"]
    )

    df = add_rolling_features(
        df,
        sensors=ROLLING_SENSORS,
        windows=ROLLING_WINDOWS
    )

    latest_row = df.tail(1)

    prediction = model.predict(
        DMatrix(latest_row[MODEL_FEATURES])
    )

    return float(prediction[0])


def predict_all_engines(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()

    predictions = []

    for engine_id in df["unit_number"].unique():
        engine_data = df[
            df["unit_number"] == engine_id
        ].copy()

        predicted_rul = predict_rul(engine_data, model)

        predictions.append({
            "unit_number": engine_id,
            "predicted_RUL": predicted_rul
        })

    return pd.DataFrame(predictions)

def predict_engine_from_db(engine_id: int) -> float:
    engine_data = get_engine_history(engine_id)

    # Convert database column names to the names expected by the model
    engine_data = engine_data.rename(columns={
        "engine_id": "unit_number",
        "cycle": "time_cycles"
    })

    model = load_model()

    predicted_rul = predict_rul(engine_data, model)

    save_prediction(engine_id, predicted_rul)

    return predicted_rul



def simulate_engine_history(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Simulate sequential RUL predictions for one engine.
    Uses only readings available up to each cycle.
    Does not save predictions to the database.
    """
    df = df.copy()

    # Accept either database column names or model column names
    df = df.rename(columns={
        "engine_id": "unit_number",
        "cycle": "time_cycles"
    })

    if df.empty:
        raise ValueError("Engine history is empty.")

    if df["unit_number"].nunique() != 1:
        raise ValueError("Pass data for exactly one engine.")

    df = df.sort_values("time_cycles").reset_index(drop=True)

    if model is None:
        model = load_model()

    engine_id = int(df["unit_number"].iloc[0])
    results = []

    for cycle in df["time_cycles"].unique():
        # Use only readings available up to this cycle
        available_data = df[
            df["time_cycles"] <= cycle
        ].copy()

        predicted_rul = predict_rul(available_data, model)

        results.append({
            "engine_id": engine_id,
            "cycle": int(cycle),
            "predicted_rul": predicted_rul
        })

    return pd.DataFrame(results)
