import pickle
import re
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
# ─── Configuration ──────────────────────────────────────────────────────
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
        pipeline_path = os.path.join(os.path.realpath(os.path.dirname(__file__)),'..','..','..',"AAPL_pipeline.pkl")
        with open(pipeline_path, 'rb') as f:
            pip = pickle.load(f)
        pip['vectorizer'] = self.vectorizer
        self.pipeline = pip
   

def tune_classifier(self):
    if self.merged_data is None:
        raise RuntimeError("Merged data not available. Run fetch + merge first.")

    # Prepare features and labels
    feats = ['Open','High','Low','Close','Volume','return','sentiment']
    X = self.merged_data[feats]
    y = (self.merged_data['next_return'] > 0).astype(int)

    # Time-based train-test split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Define param grid
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [1, 2, 3]  # useful for imbalance
    }

    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

    # Use TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(model, param_distributions=param_dist, 
                                n_iter=10, scoring='f1', cv=tscv, verbose=1, random_state=42)

    search.fit(X_train_s, y_train)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test_s)
    print("\nBest Parameters:", search.best_params_)
    print("\nClassification Report:")
    print(classification_report(y_test, preds, digits=2))

    self.pipeline = {
        'clf': best_model,
        'scaler': scaler,
        'vectorizer': self.vectorizer
    }

    with open(f'{self.symbol}_tuned_classifier.pkl', 'wb') as f:
        pickle.dump(self.pipeline, f)
    print(f"\nTuned classifier pipeline saved to {self.symbol}_tuned_classifier.pkl")


    def predict_impact(self, headline: str) -> dict:
        if self.pipeline is None:
            raise RuntimeError('Pipeline not loaded')
        clean = self._clean_text(headline)
        words = clean.split()
        neg_verbs = {'remove','removes','removed','lift','lifts','lifted',
                     'cut','cuts','cutting','ease','eases','eased',
                     'reduce','reduces','reduced','lower','lowers','lowered'}
        has_neg = any(v in words for v in neg_verbs)
        vec = self.vectorizer.transform([clean]).toarray()[0]
        features = self.vectorizer.get_feature_names_out()
        wsum = 0.0; wtot = 0.0
        for i, tf in enumerate(vec):
            term = features[i]
            if tf>0 and term in self.financial_keywords:
                w = self.financial_keywords[term]
                if has_neg and w<0:
                    w = -w
                wsum += tf*w; wtot += tf
        tfidf_score = wsum/wtot if wtot else 0.0
        fin = []
        for w in words:
            key = w[:-1] if w.endswith('s') else w
            if key in self.financial_keywords:
                coeff = self.financial_keywords[key]
                if has_neg and coeff<0:
                    coeff = -coeff
                fin.append(coeff)
        fin_score = sum(fin)/len(fin) if fin else 0.0
        adj = 0.3*tfidf_score + 0.7*fin_score
        cat = 'POSITIVE' if adj>0.1 else 'NEGATIVE' if adj< -0.1 else 'NEUTRAL'
        mag = abs(adj)
        impact = 'MINIMAL' if mag<0.05 else 'SMALL' if mag<0.1 else 'MODERATE' if mag<0.2 else 'SIGNIFICANT'
        last = self.merged_data.iloc[-1]
        vals = [last[c] for c in ['Open','High','Low','Close','Volume','return']] + [adj]
        Xp = np.array(vals).reshape(1,-1)
        Xps = self.pipeline['scaler'].transform(Xp)
        pr = float(self.pipeline['reg'].predict(Xps)[0])
        # Determine movement from regression sign
        direction = 'UP' if pr > 0 else 'DOWN'
        hits = []
        for w in words:
            key = w[:-1] if w.endswith('s') else w
            if key in self.financial_keywords:
                c = self.financial_keywords[key]
                if has_neg and c<0:
                    c = -c
                hits.append((key,c))
        sig = sorted(hits,key=lambda x:abs(x[1]),reverse=True)[:3]
        print("\nANALYSIS RESULTS:")
        print("----------------------------")
        print(f"TF-IDF SENTIMENT SCORE:     {tfidf_score:+.4f}")
        print(f"FINANCIAL CONTEXT SCORE:    {fin_score:+.4f}")
        print(f"ADJUSTED SENTIMENT SCORE:   {adj:+.4f} ({cat})")
        print(f"IMPACT ASSESSMENT:          {impact}")
        print(f"PREDICTED RETURN:           {pr*100:+.2f}%")
        print(f"PREDICTED MOVEMENT:         {direction}\n")
        print("Significant Terms:")
        return { 'tfidf_score':tfidf_score, 'financial_score':fin_score, 'adjusted_score':adj, 'category':cat, 'impact':impact, 'predicted_return':pr, 'predicted_movement':direction, 'terms':sig }
