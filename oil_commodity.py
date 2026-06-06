import yfinance as yf
import pandas as pd
import os

tickers = {
    "WTI": "CL=F",
    "USD_INDEX": "DX-Y.NYB",
    "BRENT": "BZ=F"
}

interval = "1h"
filename = f"oil_intraday_{interval}.csv"

data = {}
for name, ticker in tickers.items():
    df = yf.download(ticker, period="15d", interval=interval)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = [f"{name}_{col}" for col in df.columns]
    data[name] = df

new_df = pd.concat(data.values(), axis=1).dropna()
new_df.index.name = "Datetime"

if os.path.exists(filename):
    existing_df = pd.read_csv(filename, index_col="Datetime", parse_dates=True)
    combined = pd.concat([existing_df, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]  # garde la plus récente
    combined = combined.sort_index()
    print(f"✅ Existant: {len(existing_df)} lignes | Nouvelles: {len(new_df)} lignes | Final: {len(combined)} lignes")
else:
    combined = new_df.sort_index()
    print(f"✅ Nouveau fichier créé: {len(combined)} lignes")

combined.to_csv(filename)
print(combined.tail(5))