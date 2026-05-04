import yfinance as yf
import pandas as pd

tickers = {
    "WTI": "CL=F",
    "USD_INDEX": "DX-Y.NYB",
    "BRENT": "BZ=F"
}


interval = "1h" 

data = {}

for name, ticker in tickers.items():
    df = yf.download(
        ticker,
        period="10d",      
        interval=interval
    )

    df = df[["Open", "High", "Low", "Close", "Volume"]]

    df.columns = [f"{name}_{col}" for col in df.columns]

    data[name] = df

final_df = pd.concat(data.values(), axis=1)

final_df = final_df.dropna()

final_df.to_csv(f"oil_intraday_{interval}.csv")
