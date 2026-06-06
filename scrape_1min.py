import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def scrape_wti_1min():
    print("⏳ Téléchargement des données WTI 1 minute (7 derniers jours)...")
    
    # Téléchargement WTI, Brent et USD Index en 1 minute
    tickers = {
        "WTI"      : "CL=F",
        "BRENT"    : "BZ=F",
        "USD_INDEX": "DX-Y.NYB"
    }
    
    data = {}
    for name, ticker in tickers.items():
        print(f"   📥 Téléchargement {name} ({ticker})...")
        df = yf.download(
            ticker,
            period   = "7d",      # 7 derniers jours
            interval = "1m",      # fréquence 1 minute
            progress = False
        )
        df.columns = [f"{name}_{col}" for col in df.columns]
        data[name] = df
    
    # Fusion des trois tickers
    final_df = pd.concat(data.values(), axis=1, sort=True)
    final_df = final_df.dropna()
    final_df.index.name = "Datetime"
    final_df = final_df.reset_index()
    
    print(f"\n✅ {len(final_df)} lignes récupérées")
    print(f"   Du : {final_df['Datetime'].min()}")
    print(f"   Au : {final_df['Datetime'].max()}")
    print(final_df.tail(3))
    
    # Sauvegarde
    final_df.to_csv("oil_intraday_1m.csv", index=False)
    print("\n✅ Sauvegardé dans oil_intraday_1m.csv")
    
    return final_df

df = scrape_wti_1min()