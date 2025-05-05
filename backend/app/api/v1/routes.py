from app.services.predictor import get_best_stock, get_sentiment
from fastapi import APIRouter
from app.schemas.stock import PredictionRequest, PredictionResponse,SentimentRequest, SentimentResponse
from typing import List


router = APIRouter()

@router.post("/predict", response_model=List[PredictionResponse])
def predict_stock(data: PredictionRequest):
    return get_best_stock(data)

@router.post("/predict-sentiment",response_model=SentimentResponse)
def predict_from_sentiment(sentiment: SentimentRequest):
    return get_sentiment(sentiment)  # assuming model takes raw sentiment text