from pydantic import BaseModel
from typing import List,Tuple

class PredictionRequest(BaseModel):
    tickers: List[str]
    date: str # might add datetime
    
class PredictionResponse(BaseModel):
    ticker: str
    predicted_return: float
    predicted_return_percent: float
    current_price: float
    prediction_date:str
    recommendation: str
    
class SentimentRequest(BaseModel):
    ticker:str
    headline:str
    
class SentimentResponse(BaseModel):
    tfidf_score:float
    financial_score:float
    adjusted_score: float
    category: str
    impact: str
    predicted_return:float
    predicted_movement:str
    terms:List[Tuple[str,float]]
    
    