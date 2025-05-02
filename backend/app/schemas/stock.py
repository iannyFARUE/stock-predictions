from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    tickers: List[str]
    date: str # might add datetime
    
class PredictionResponse(BaseModel):
    recommended_ticker: str
    expected_return: float
    rankings: List[dict]
    
    