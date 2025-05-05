#!/usr/bin/env python3

import pickle
import re
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier

# ─── Configuration ───────────────────────────────────────────────────────
NEWSAPI_KEY = "f22a6da5ffb24fd7ab7d51b12a3a1885"
SYMBOL      = "AAPL"
PIPELINE_FN = f"AAPL_pipeline.pkl"
#PIPELINE_FN = f"{SYMBOL}_pipeline.pkl"
# ──────────────────────────────────────────────────────────────────────────

class StockNewsSentimentTool:
    def __init__(self, api_key, symbol):
        self.api_key    = api_key
        self.symbol     = symbol
        # must match training vectorizer settings
        self.vectorizer = TfidfVectorizer(max_features=200,
                                          stop_words='english',
                                          ngram_range=(1,2),
                                          min_df=2)
        # same keywords as training
        self.financial_keywords = {
            'profit': 0.8, 'loss': -0.8, 'growth': 0.7, 'risk': -0.5,
            'beat': 0.8, 'miss': -0.8, 'iphone': 0.5,
            'tariff': -0.7, 'tariffs': -0.7, 'delay': -0.6,
            'remove': 0.7, 'removes': 0.7, 'removed': 0.7,
            'lift': 0.6, 'lifts': 0.6, 'lifted': 0.6,
            'cut': 0.6, 'cuts': 0.6, 'cutting': 0.6,
            'ease': 0.6, 'eases': 0.6, 'eased': 0.6,
            'reduce': 0.7, 'reduces': 0.7, 'reduced': 0.7,
            'lower': 0.6, 'lowers': 0.6, 'lowered': 0.6,
            '30%': 0.9, '-30%': -0.9
        }
        self.news_data   = None
        self.stock_data  = None
        self.merged_data = None
        self.pipeline    = None

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def fetch_news(self, days=30):
        end = datetime.now()
        start = end - timedelta(days=days)
        resp = requests.get(
            'https://newsapi.org/v2/everything',
            params={
                'q': f"{self.symbol} stock",
                'from': start.strftime('%Y-%m-%d'),
                'to': end.strftime('%Y-%m-%d'),
                'language': 'en', 'sortBy': 'publishedAt',
                'apiKey': self.api_key
            }
        )
        articles = resp.json().get('articles', []) if resp.status_code == 200 else []
        if not articles:
            dates = [end.date() - timedelta(days=i) for i in range(days)]
            self.news_data = pd.DataFrame({'date': dates,
                                           'sentiment': np.random.uniform(-0.5, 0.5, days)})
        else:
            df = pd.DataFrame(articles)
            df['date'] = pd.to_datetime(df['publishedAt']).dt.date
            df['text'] = (df['title'].fillna('') + ' ' + df['description'].fillna('')).apply(self._clean_text)
            self.vectorizer.fit(df['text'])
            tfidf_mat = self.vectorizer.transform(df['text'])
            feats = self.vectorizer.get_feature_names_out()
            scores = []
            for arr in tfidf_mat.toarray():
                weighted_sum = 0.0
                total_w = 0.0
                for i, tfidf in enumerate(arr):
                    term = feats[i]
                    if tfidf > 0 and term in self.financial_keywords:
                        weighted_sum += tfidf * self.financial_keywords[term]
                        total_w += tfidf
                scores.append(weighted_sum / total_w if total_w else 0.0)
            df['sentiment'] = scores
            self.news_data = df.groupby('date', as_index=False)['sentiment'].mean()

    def fetch_stock(self, days=30):
        end = datetime.now()
        df = yf.Ticker(self.symbol).history(
            start=(end - timedelta(days=days+5)).strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d')
        ).reset_index()
        df['date'] = df['Date'].dt.date
        df['return'] = df['Close'].pct_change()
        df['next_return'] = df['return'].shift(-1)
        df.fillna(method='bfill', inplace=True)
        self.stock_data = df[['date','Open','High','Low','Close','Volume','return','next_return']]

    def merge_data(self):
        if self.stock_data is None or self.news_data is None:
            raise RuntimeError("Must fetch both stock and news first.")
        self.merged_data = pd.merge(self.stock_data, self.news_data, on='date', how='left').fillna(0)

    def load_pipeline(self):
        with open(PIPELINE_FN, 'rb') as f:
            pip = pickle.load(f)
        pip['vectorizer'] = self.vectorizer
        self.pipeline = pip

    def predict_impact(self, headline):
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded.")
        clean = self._clean_text(headline)
        words = clean.split()
        vec = self.pipeline['vectorizer'].transform([clean]).toarray()[0]
        feats = self.pipeline['vectorizer'].get_feature_names_out()
        weighted_sum = total_w = 0.0
        for i, tfidf in enumerate(vec):
            term = feats[i]
            if tfidf > 0 and term in self.financial_keywords:
                w = self.financial_keywords[term]
                if any(v in words for v in ['remove','reduce','lift','cut','ease']) and w < 0:
                    w = -w
                weighted_sum += tfidf * w
                total_w += tfidf
        tfidf_score = weighted_sum / total_w if total_w else 0.0
        fin = []
        for w in words:
            key = w[:-1] if w.endswith('s') else w
            if key in self.financial_keywords:
                coeff = self.financial_keywords[key]
                if any(v in words for v in ['remove','reduce','lift','cut','ease']) and coeff < 0:
                    coeff = -coeff
                fin.append(coeff)
        fin_score = float(sum(fin) / len(fin)) if fin else 0.0
        adj = 0.3 * tfidf_score + 0.7 * fin_score
        cat = 'POSITIVE' if adj > 0.1 else 'NEGATIVE' if adj < -0.1 else 'NEUTRAL'
        mag = abs(adj)
        impact = 'MINIMAL' if mag < 0.05 else 'SMALL' if mag < 0.1 else 'MODERATE' if mag < 0.2 else 'SIGNIFICANT'
        # movement based on adjusted sentiment
        direction = 'UP' if adj > 0 else 'DOWN'
        # print results
        print("\nANALYSIS RESULTS:")
        print("----------------------------")
        
        print(f"ADJUSTED SENTIMENT SCORE: {adj:+.4f} ({cat})")
        print(f"IMPACT ASSESSMENT:        {impact}")
        print(f"PREDICTED MOVEMENT:       {direction}\n")

if __name__ == '__main__':
    tool = StockNewsSentimentTool(NEWSAPI_KEY, SYMBOL)
    print(f"Loading pipeline {PIPELINE_FN} and data...")
    tool.load_pipeline()
    tool.fetch_stock(30)
    tool.fetch_news(30)
    tool.merge_data()
    print("\n--- Enter headline (type 'exit' to quit) ---")
    while True:
        h = input('> ').strip()
        if h.lower() in ('exit','quit'):
            print("Goodbye!")
            break
        tool.predict_impact(h)
