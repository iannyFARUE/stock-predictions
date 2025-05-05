from app.services.predictor import get_best_stock
from fastapi import APIRouter
from app.schemas.stock import PredictionRequest, PredictionResponse
from typing import List


router = APIRouter()

@router.post("/predict", response_model=List[PredictionResponse])
def predict_stock(data: PredictionRequest):
    return get_best_stock(data)