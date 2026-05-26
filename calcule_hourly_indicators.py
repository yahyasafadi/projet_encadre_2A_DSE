import pandas as pd
import numpy as np
from transformers import pipeline as hf_pipeline
import warnings
warnings.filterwarnings('ignore')

# ── CHARGEMENT ──────────────────────────────────────────────────
df = pd.read_csv('data_merged.csv', index_col=0, parse_dates=['Datetime'])
df = df.set_index('Datetime')

# Renommage des colonnes (incluant les nouvelles colonnes demandées)
df = df.rename(columns={
    "WTI_('Open', 'CL=F')"              : 'WTI_Open',
    "WTI_('High', 'CL=F')"              : 'WTI_High',
    "WTI_('Low', 'CL=F')"               : 'WTI_Low',
    "WTI_('Close', 'CL=F')"             : 'WTI_Close',
    "WTI_('Volume', 'CL=F')"            : 'WTI_Volume',
    "USD_INDEX_('Open', 'DX-Y.NYB')"    : 'USD_Open',
    "USD_INDEX_('Close', 'DX-Y.NYB')"   : 'USD_Close',
    "BRENT_('Close', 'BZ=F')"           : 'BRENT_Close',
})

# ── FEATURE ENGINEERING ─────────────────────────────────────────
# (Calcul des indicateurs techniques comme dans votre script d'origine)
sma20 = df['WTI_Close'].rolling(20).mean()
df['Close_SMA20_ratio'] = df['WTI_Close'] / sma20

ema12 = df['WTI_Close'].ewm(span=12, adjust=False).mean()
ema26 = df['WTI_Close'].ewm(span=26, adjust=False).mean()
df['EMA_ratio'] = ema12 / ema26

delta = df['WTI_Close'].diff()
gain  = delta.where(delta > 0, 0).rolling(14).mean()
loss  = (-delta).where(delta < 0, 0).rolling(14).mean()
df['RSI_14'] = 100 - (100 / (1 + gain / loss))

df['return_12h'] = df['WTI_Close'].pct_change(12)
df['Spread_Brent_WTI'] = df['BRENT_Close'] - df['WTI_Close']
df['Corr_Brent_USD_30'] = df['BRENT_Close'].rolling(30).corr(df['USD_Close'])

prev_close = df['WTI_Close'].shift(1)
true_range = pd.concat([
    df['WTI_High'] - df['WTI_Low'],
    (df['WTI_High'] - prev_close).abs(),
    (df['WTI_Low']  - prev_close).abs(),
], axis=1).max(axis=1)
atr14 = true_range.rolling(14).mean()
atr48 = true_range.rolling(48).mean()
atr_ratio = atr14 / (atr48 + 1e-9)
df['ADX_proxy'] = atr_ratio.rolling(14).mean()

df['session_europe'] = ((df.index.hour >= 9) & (df.index.hour < 15)).astype(int)

macd = ema12 - ema26
macd_signal = macd.ewm(span=9, adjust=False).mean()
macd_hist = macd - macd_signal
df['MACD_slope'] = macd_hist.diff(3)

# ── SENTIMENT ───────────────────────────────────────────────────
# (Pipeline FinBERT - gardé tel quel)
sentiment_pipe = hf_pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)

def get_sentiment_score(text):
    if pd.isna(text) or str(text).strip() in ('no_news_available', ''): return np.nan
    headlines = [h.strip() for h in str(text).split('|') if h.strip()]
    if not headlines: return np.nan
    try:
        results = sentiment_pipe(headlines, truncation=True, max_length=512)
        scores = [r['score'] if r['label'] == 'positive' else -r['score'] if r['label'] == 'negative' else 0.0 for r in results]
        return float(np.mean(scores))
    except: return np.nan

df['sentiment_score'] = df['title'].apply(get_sentiment_score)
df['sentiment_score'] = df['sentiment_score'].ffill().fillna(0.0)

# ── EXPORT ───────────────────────────────────────────────────────
# Liste mise à jour sans 'target' et avec les données brutes demandées
FEATURES = [
    'WTI_Open', 'WTI_High', 'WTI_Low', 'WTI_Close', 'WTI_Volume', 'USD_Open',
    'Close_SMA20_ratio', 'EMA_ratio', 'RSI_14', 'return_12h', 'Spread_Brent_WTI', 
    'Corr_Brent_USD_30', 'ADX_proxy', 'session_europe', 'MACD_slope', 'sentiment_score'
]

df_model = df[FEATURES].dropna()
df_model.to_csv('wti_features_v3.csv')