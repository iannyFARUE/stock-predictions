from backend.app.services.predictor import get_best_stock
from fastapi import APIRouter
from app.schemas.stock import PredictionRequest, PredictionResponse


router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict_stock(data: PredictionRequest):
    return get_best_stock(data)