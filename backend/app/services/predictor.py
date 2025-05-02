import pickle
import pandas as pd
from app.schemas.stock import PredictionRequest, PredictionResponse

model = pickle.load(open("models/best_model.pkl", "rb"))

# Dummy implementation
def get_best_stock(data: PredictionRequest):
    sample = pd.DataFrame({"ticker": data.tickers, "5d_return": [0.02]*len(data.tickers)})
    sample['predicted_return'] = model.predict([[0.02]*8 for _ in data.tickers])
    top = sample.loc[sample['predicted_return'].idxmax()]
    rankings = sample.to_dict(orient='records')
    return PredictionResponse(recommended_ticker=top['ticker'], expected_return=top['predicted_return'], rankings=rankings)