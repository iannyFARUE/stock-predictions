from pydantic import BaseModel
from typing import List

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
    
    