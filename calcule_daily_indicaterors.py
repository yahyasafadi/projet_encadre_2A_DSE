import pandas as pd
import numpy as np

def clean_and_feature_engineering(file_path):
    # 1. Chargement et préparation de base
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date").reset_index(drop=True)
    
    # Remplissage des valeurs manquantes
    cols_price = ["Open", "High", "Low", "Close", "Volume"]
    df[cols_price] = df[cols_price].ffill()

    # 2. Indicateurs techniques optimisés
    # Tendance : Rapport de positionnement (plus robuste que la valeur brute)
    sma200 = df["Close"].rolling(200).mean()
    sma50 = df["Close"].rolling(50).mean()
    df["price_vs_SMA200"] = (df["Close"] - sma200) / sma200
    df["SMA50_vs_SMA200"] = (sma50 - sma200) / sma200

    # Momentum : RSI (14 périodes)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1/14).mean()
    df["RSI14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # Volatilité : ATR normalisé et Volatilité historique
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-df["Close"].shift(1)).abs(), (df["Low"]-df["Close"].shift(1)).abs()], axis=1).max(axis=1)
    df["ATR14_pct"] = tr.ewm(alpha=1/14).mean() / df["Close"]
    df["hvol20"] = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)

    # Volume : Ratio de volume (pics d'activité)
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # MACD : Histogramme (changement de momentum)
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    df["MACD_hist"] = macd - macd.ewm(span=9).mean()

    # 3. Nettoyage final
    cols_to_keep = ["Date", "price_vs_SMA200", "SMA50_vs_SMA200", "RSI14", 
                    "ATR14_pct", "hvol20", "vol_ratio", "MACD_hist"]
    
    return df[cols_to_keep].dropna()

# Exécution
df_clean = clean_and_feature_engineering("crude_oil_wti_daily_last_5_years.csv")
df_clean.to_csv("daily_indicators.csv", index=False)
