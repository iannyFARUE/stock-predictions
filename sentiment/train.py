
import pickle
import requests
import pandas as pd
import numpy as np
import re
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────
NEWSAPI_KEY = "f22a6da5ffb24fd7ab7d51b12a3a1885"
SYMBOL      = "MSFT"
# ──────────────────────────────────────────────────────────────────────────

class StockNewsSentimentTool:
    def __init__(self, api_key: str, symbol: str):
        self.api_key    = api_key
        self.symbol     = symbol
        # TF-IDF vectorizer for news
        self.vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            ngram_range=(1,2),
            min_df=2
        )
        # Financial keyword weights, including negation verbs
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

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def fetch_news(self, days: int = 30) -> pd.DataFrame:
        end = datetime.now()
        start = end - timedelta(days=days)
        params = {'q': f"{self.symbol} stock", 'from': start.strftime('%Y-%m-%d'),
                  'to': end.strftime('%Y-%m-%d'), 'language': 'en',
                  'sortBy': 'publishedAt', 'apiKey': self.api_key}
        resp = requests.get('https://newsapi.org/v2/everything', params=params)
        articles = resp.json().get('articles', []) if resp.status_code == 200 else []
        if not articles:
            dates = [end.date() - timedelta(days=i) for i in range(days)]
            df = pd.DataFrame({'date': dates, 'text': ['']*days,
                               'sentiment': np.random.uniform(-0.5,0.5,days)})
        else:
            df = pd.DataFrame(articles)
            df['date'] = pd.to_datetime(df['publishedAt']).dt.date
            df['text'] = (df['title'].fillna('')+' '+df['description'].fillna('')) \
                        .apply(self._clean_text)
            self.vectorizer.fit(df['text'])
            tfidf_mat = self.vectorizer.transform(df['text'])
            features = self.vectorizer.get_feature_names_out()
            scores = []
            for row in tfidf_mat:
                arr = row.toarray()[0]
                weighted_sum = 0.0
                weight_total = 0.0
                for i, tfidf in enumerate(arr):
                    term = features[i]
                    if tfidf > 0 and term in self.financial_keywords:
                        weighted_sum += tfidf * self.financial_keywords[term]
                        weight_total += tfidf
                scores.append(weighted_sum/weight_total if weight_total else 0.0)
            df['sentiment'] = scores
        daily = df.groupby('date', as_index=False)['sentiment'].mean()
        self.news_data = daily
        return daily

    def fetch_stock(self, days: int = 30) -> pd.DataFrame:
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
        return self.stock_data

    def merge_data(self) -> pd.DataFrame:
        if self.news_data is None or self.stock_data is None:
            raise RuntimeError('Fetch news and stock first')
        merged = pd.merge(self.stock_data, self.news_data,
                          on='date', how='left').fillna(0)
        self.merged_data = merged
        return merged

    # inside your StockNewsSentimentTool.train_pipeline()

# After merge_data, ensure merged_data has 'sentiment' column
# Then in prepare_features:
    def prepare_features(self):
        m = self.merged_data
        # Include 'sentiment' when you build X
        feats = ['Open','High','Low','Close','Volume','return','sentiment']
        X = m[feats]
        y_ret = m['next_return']
        y_mov = (y_ret > 0).astype(int)
        return X, y_ret, y_mov


    def train_pipeline(self):
        X, y_ret, y_mov = self.prepare_features()
        Xtr, Xva, ytr_ret, yva_ret = train_test_split(X, y_ret,
                                                    test_size=0.2, shuffle=False)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
        reg = XGBRegressor(random_state=42)
        reg.fit(Xtr_s, ytr_ret, eval_set=[(Xva_s, yva_ret)], verbose=False)
        preds = reg.predict(Xva_s)
        print('Return Model Evaluation:')
        print(f'  MSE: {mean_squared_error(yva_ret,preds):.6f}')
        print(f'  R2 : {r2_score(yva_ret,preds):.6f}\n')
        _, _, ytr_mov, yva_mov = train_test_split(X, y_mov,
                                                  test_size=0.2, shuffle=False)
        clf = XGBClassifier(random_state=42)
        clf.fit(Xtr_s, ytr_mov, eval_set=[(Xva_s,yva_mov)], verbose=False)
        print('Movement Model Evaluation:')
        print(classification_report(yva_mov,clf.predict(Xva_s),digits=2))
        self.pipeline = {'reg':reg,'clf':clf,'scaler':scaler}
        with open(f'{self.symbol}_pipeline.pkl','wb') as f:
            pickle.dump(self.pipeline,f)
        print(f'\nPipeline saved to {self.symbol}_pipeline.pkl\n')

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
        print(f"PREDICTED MOVEMENT:         {dir}\n")
        print("Significant Terms:")
        for t,s in sig:
            print(f"  {t}: {s:+.4f}")
        print("\nReady for next analysis...\n")
        return { 'tfidf_score':tfidf_score, 'financial_score':fin_score, 'adjusted_score':adj, 'category':cat, 'impact':impact, 'predicted_return':pr, 'predicted_movement':dir, 'terms':sig }

if __name__ == '__main__':
    tool = StockNewsSentimentTool(NEWSAPI_KEY, SYMBOL)
    print(f"\n--- Fetching data for {SYMBOL} ---")
    tool.fetch_stock(days=30)
    tool.fetch_news(days=30)
    tool.merge_data()
    print("\n--- Training models ---")
    tool.train_pipeline()
    #print("\n--- Enter news headline (type 'exit' to quit) ---")
    #while True:
    #    txt = input('> ').strip()
    #    if txt.lower() in ('exit','quit'): break
    #    tool.predict_impact(txt)
