import uuid

from db.queries import (
    get_engine_ids,
    get_latest_predictions,
    get_engine_history,
    get_prediction_history as db_get_prediction_history,
    get_simulation_predictions,
    save_simulation_predictions,
)
from ml.intervals import add_interval
from ml.predict import MODEL_TYPE, predict_engine_from_db, load_model, simulate_engine_history
from api.services.alerts import get_maintenance_status

from config import DATA_DIR
import pandas as pd

def get_latest_predictions_with_status():
    rows = get_latest_predictions()

    return [
        {
            "id": row[0],
            "engine_id": row[1],
            "predicted_rul": row[2],
            "prediction_time": row[3],
            "maintenance_status": get_maintenance_status(float(row[2])),
            **add_interval(float(row[2]), MODEL_TYPE),
        }
        for row in rows
    ]


def get_prediction_history(engine_id: int):
    rows = db_get_prediction_history(engine_id)

    return [
        {
            "id": row[0],
            "engine_id": row[1],
            "predicted_rul": row[2],
            "prediction_time": row[3],
        }
        for row in rows
    ]


def create_prediction(engine_id: int):
    history = get_engine_history(engine_id)

    if history.empty:
        return None

    predicted_rul = predict_engine_from_db(engine_id)
    status = get_maintenance_status(predicted_rul)

    return {
        "engine_id": engine_id,
        "predicted_rul": predicted_rul,
        "maintenance_status": status,
        **add_interval(predicted_rul, MODEL_TYPE),
    }


def run_all_predictions():
    results = []

    for engine_id in get_engine_ids():
        predicted_rul = predict_engine_from_db(engine_id)

        results.append({
            "engine_id": engine_id,
            "predicted_rul": predicted_rul,
            "maintenance_status": get_maintenance_status(predicted_rul),
        })

    return {
        "engines_predicted": len(results),
        "critical": sum(r["maintenance_status"] == "CRITICAL" for r in results),
        "warning": sum(r["maintenance_status"] == "WARNING" for r in results),
        "normal": sum(r["maintenance_status"] == "NORMAL" for r in results),
    }


def get_simulation_results(simulation_id: str):
    rows = get_simulation_predictions(simulation_id)

    return [
        {**row, **add_interval(float(row["predicted_rul"]), MODEL_TYPE)}
        for row in rows
    ]



def run_simulation(engine_id: int):
    engine_data = get_engine_history(engine_id)

    if engine_data.empty:
        return None

    rul_values = pd.read_csv(DATA_DIR / "RUL_FD001.txt", header=None)
    actual_rul = float(rul_values.iloc[engine_id - 1, 0])

    simulation_id = uuid.uuid4()
    model = load_model()
    simulation_df = simulate_engine_history(engine_data, model=model)

    save_simulation_predictions(
        simulation_id=simulation_id,
        engine_id=engine_id,
        simulation_df=simulation_df,
        actual_rul=actual_rul,
    )

    return {
        "simulation_id": str(simulation_id),
        "engine_id": engine_id,
        "cycles_simulated": len(simulation_df),
    }