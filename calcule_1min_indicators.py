import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── CHARGEMENT ───────────────────────────────────────────────────
print("⏳ Chargement des données 1 minute + news...")
df = pd.read_csv('data_merged_1min.csv')

# Renommage automatique des colonnes
rename = {}
for col in df.columns:
    if 'Datetime' in col or 'datetime' in col or 'timestamp' in col:
        rename[col] = 'Datetime'
    elif 'Close' in col and 'WTI' in col:
        rename[col] = 'WTI_Close'
    elif 'Open' in col and 'WTI' in col:
        rename[col] = 'WTI_Open'
    elif 'High' in col and 'WTI' in col:
        rename[col] = 'WTI_High'
    elif 'Low' in col and 'WTI' in col:
        rename[col] = 'WTI_Low'
    elif 'Volume' in col and 'WTI' in col:
        rename[col] = 'WTI_Volume'
    elif 'Close' in col and 'BRENT' in col:
        rename[col] = 'BRENT_Close'
    elif 'Close' in col and 'USD' in col:
        rename[col] = 'USD_Close'

df = df.rename(columns=rename)
df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
df = df.sort_values('Datetime').reset_index(drop=True)
print(f"✅ {len(df)} lignes chargées")
print(f"   Colonnes : {list(df.columns)}")

# ── FEATURES TECHNIQUES ──────────────────────────────────────────
# SMA 20 minutes
sma20 = df['WTI_Close'].rolling(20).mean()
df['Close_SMA20_ratio'] = df['WTI_Close'] / sma20

# EMA 12 et 26 minutes
ema12 = df['WTI_Close'].ewm(span=12, adjust=False).mean()
ema26 = df['WTI_Close'].ewm(span=26, adjust=False).mean()
df['EMA_ratio'] = ema12 / ema26

# MACD
macd             = ema12 - ema26
macd_signal      = macd.ewm(span=9, adjust=False).mean()
df['MACD_slope'] = (macd - macd_signal).diff(3)

# RSI 14 minutes
delta        = df['WTI_Close'].diff()
gain         = delta.where(delta > 0, 0).rolling(14).mean()
loss         = (-delta).where(delta < 0, 0).rolling(14).mean()
df['RSI_14'] = 100 - (100 / (1 + gain / loss))

# Rendements
df['return_12min'] = df['WTI_Close'].pct_change(12)
df['return_60min'] = df['WTI_Close'].pct_change(60)

# Spread Brent - WTI
if 'BRENT_Close' in df.columns:
    df['Spread_Brent_WTI'] = df['BRENT_Close'] - df['WTI_Close']
else:
    df['Spread_Brent_WTI'] = 0

# Corrélation Brent/USD sur 30 minutes
if 'BRENT_Close' in df.columns and 'USD_Close' in df.columns:
    df['Corr_Brent_USD_30'] = df['BRENT_Close'].rolling(30).corr(df['USD_Close'])
else:
    df['Corr_Brent_USD_30'] = 0

# ATR & ADX proxy
prev_close = df['WTI_Close'].shift(1)
tr = pd.concat([
    df['WTI_High'] - df['WTI_Low'],
    (df['WTI_High'] - prev_close).abs(),
    (df['WTI_Low']  - prev_close).abs()
], axis=1).max(axis=1)
atr14           = tr.rolling(14).mean()
atr48           = tr.rolling(48).mean()
df['ADX_proxy'] = (atr14 / (atr48 + 1e-9)).rolling(14).mean()

# Volatilité 20 minutes
df['volatility_20min'] = df['WTI_Close'].pct_change().rolling(20).std()

# Sessions de trading
df['session_europe'] = (
    (df['Datetime'].dt.hour >= 9) &
    (df['Datetime'].dt.hour < 15)
).astype(int)

df['session_us'] = (
    (df['Datetime'].dt.hour >= 14) &
    (df['Datetime'].dt.hour < 21)
).astype(int)

# Features cycliques d'heure
df['hour_sin'] = np.sin(2 * np.pi * df['Datetime'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['Datetime'].dt.hour / 24)

# ── SENTIMENT FINBERT (BATCH) ────────────────────────────────────
if 'title' in df.columns:
    print("⏳ Chargement FinBERT...")
    from transformers import pipeline as hf_pipeline
    sentiment_pipe = hf_pipeline(
        "sentiment-analysis",
        model     = "ProsusAI/finbert",
        device    = -1,
        batch_size= 32
    )

    # On calcule le sentiment uniquement sur les heures uniques (pas chaque minute)
    df_unique = df[['Datetime', 'title']].copy()
    df_unique['hour'] = df_unique['Datetime'].dt.floor('h')
    df_hours = df_unique.drop_duplicates(subset='hour')[['hour', 'title']].reset_index(drop=True)

    print(f"   📰 Analyse de {len(df_hours)} heures uniques avec FinBERT...")

    scores = []
    for text in df_hours['title']:
        if pd.isna(text) or str(text).strip() in ('no_news_available', ''):
            scores.append(0.0)
        else:
            headlines = [h.strip()[:512] for h in str(text).split('|') if h.strip()]
            if not headlines:
                scores.append(0.0)
            else:
                try:
                    results = sentiment_pipe(headlines[:3], truncation=True, max_length=512)
                    s = [r['score'] if r['label']=='positive' else
                        -r['score'] if r['label']=='negative' else 0.0
                        for r in results]
                    scores.append(float(np.mean(s)))
                except:
                    scores.append(0.0)

    df_hours['sentiment_score'] = scores

    # Merge du sentiment sur toutes les minutes
    df['hour'] = df['Datetime'].dt.floor('h')
    df = df.merge(df_hours[['hour', 'sentiment_score']], on='hour', how='left')
    df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
    df = df.drop(columns=['hour'])
    print("✅ Sentiment calculé !")
else:
    df['sentiment_score'] = 0.0

# ── TARGET ───────────────────────────────────────────────────────
df['target'] = (df['WTI_Close'].shift(-1) > df['WTI_Close']).astype(int)

# ── EXPORT ───────────────────────────────────────────────────────
FEATURES = [
    'Datetime', 'WTI_Close',
    'Close_SMA20_ratio', 'EMA_ratio', 'MACD_slope',
    'RSI_14', 'return_12min', 'return_60min',
    'Spread_Brent_WTI', 'Corr_Brent_USD_30',
    'ADX_proxy', 'volatility_20min',
    'session_europe', 'session_us',
    'hour_sin', 'hour_cos',
    'sentiment_score', 'target'
]

df_model = df[FEATURES].dropna()
df_model.to_csv('wti_features_1min.csv', index=False)

print(f"\n✅ {len(df_model)} lignes sauvegardées dans wti_features_1min.csv")
print(df_model.tail(3))