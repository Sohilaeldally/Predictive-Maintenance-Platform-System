from fastapi import APIRouter, HTTPException

from api.services import prediction_service

router = APIRouter(prefix="/simulations", tags=["simulations"])

from uuid import UUID

@router.get("/{simulation_id}")
def simulation_predictions(simulation_id: UUID):
    rows = prediction_service.get_simulation_results(str(simulation_id))

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No simulation results found for simulation ID {simulation_id}",
        )

    return rows

@router.post("/{engine_id}")
def create_simulation(engine_id: int):
    result = prediction_service.run_simulation(engine_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No sensor data found for engine {engine_id}",
        )

    return result