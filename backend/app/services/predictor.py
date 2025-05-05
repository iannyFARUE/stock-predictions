import pickle
import pandas as pd
from app.schemas.stock import PredictionRequest, PredictionResponse
import datetime
import yfinance as yf
import ta
import joblib
import dill
from app.core.constants import TICKERS
import os

def load_data(start_date:datetime.datetime, end_data:str):
    return yf.download(TICKERS, start=start_date, end=end_data, group_by='ticker', auto_adjust=True)
    
def prepare_stock_features(days:int = 60):
    END_DATE = '2020-10-30'
    START_DATE = datetime.datetime.strptime(END_DATE, "%Y-%m-%d") - datetime.timedelta(days=days)
    data = load_data(START_DATE,END_DATE)
    dfs = []
    for ticker in data.columns.levels[0]:
        df = data[ticker].copy()
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_20d'] = df['Close'].pct_change(20)
        df['volatility_20d'] = df['Close'].pct_change().rolling(20).std()
        df['rsi_14'] = ta.momentum.RSIIndicator(df['Close'].squeeze(), window=14).rsi()
        macd = ta.trend.MACD(df['Close'].squeeze())
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df['Close'].squeeze())
        df['bollinger_h'] = bb.bollinger_hband()
        df['bollinger_l'] = bb.bollinger_lband()
        df['ticker'] = ticker
        dfs.append(df)
    feature_df = pd.concat(dfs)
    feature_df.reset_index(inplace=True)
    return feature_df

def load_fundamentals(TICKERS:list[str]):
    fundamentals_model = os.path.join(os.path.realpath(os.path.dirname(__file__)),"..","..","..","Technical Analysis-DTK","get_fundamentals.pkl")
    with open(fundamentals_model, 'rb') as f:
        get_fundamentals = dill.load(f)
        latest_fundamentals = []

        for ticker in TICKERS:
            pe, pb = get_fundamentals(ticker)
            latest_fundamentals.append({'ticker': ticker, 'pe_ratio': pe, 'pb_ratio': pb})

        latest_fundamentals_df = pd.DataFrame(latest_fundamentals).set_index('ticker')
    return latest_fundamentals_df


            
    

# Dummy implementation
def get_best_stock(data: PredictionRequest):
    model_path = os.path.join(os.path.realpath(os.path.dirname(__file__)),"..","..","..","Technical Analysis-DTK","asset_selection_model.pkl")
    scaler_path = os.path.join(os.path.realpath(os.path.dirname(__file__)),"..","..","..","Technical Analysis-DTK","scaler.pkl")
    rf_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    latest_features = prepare_stock_features()
    latest_fundamentals = load_fundamentals(TICKERS=TICKERS)
    latest_features = latest_features.merge(latest_fundamentals, on='ticker', how='left')
    
    # --- 8. Prepare Dataset for Prediction ---
    feature_cols = [
        'return_5d', 'return_20d', 'volatility_20d', 'rsi_14', 
        'macd', 'macd_signal', 'bollinger_h', 'bollinger_l',
        'pe_ratio', 'pb_ratio'
    ]
    
    latest_features = latest_features.dropna(subset=feature_cols)
    X_latest = latest_features[feature_cols]
    X_latest_scaled = scaler.transform(X_latest)

    # --- 9. Predict Future Returns ---
    predictions = rf_model.predict(X_latest_scaled)

    latest_features['predicted_future_return'] = predictions

    # --- 10. Get Latest Date and Top 3 Stocks ---
    latest_date = latest_features['Date'].max()
    latest_data = latest_features[latest_features['Date'] == latest_date]
    latest_data['recommendation'] = latest_data['predicted_future_return'].apply(lambda x: 'Buy' if x > 0.08 else 'Sell' if x < -0.08 else 'Do not Enter')
    
    responses = []
    for _, row in latest_data.iterrows():
        current_price = row.get('Close')
        prediction_response = PredictionResponse(
            ticker=row['ticker'],
            predicted_return=row['predicted_future_return'],
            predicted_return_percent=row['predicted_future_return'] * 100,  # Convert to percentage
            current_price=current_price,
            prediction_date=latest_date.strftime('%Y-%m-%d'),  # Format the date as string
            recommendation=row['recommendation']
        )
        responses.append(prediction_response)
    
    return responses