import yfinance as yf
import pandas as pd
from datetime import datetime

# get_market function 
def get_market_monitoring():
    tickers = {
        "Brent_Oil": "BZ=F",
        "WTI_Oil": "CL=F",
        "Dollar_Index": "DX-Y.NYB"
    }
    
    results = []

    print(f"--- Collecte du {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    for name, symbol in tickers.items():
        try:
            ticker_obj = yf.Ticker(symbol)
            
          
            data = ticker_obj.history(period="1d", interval="1m")
            
            if not data.empty:
                last_row = data.iloc[-1]
                
                results.append({
                    "Indicateur": name,
                    "Symbol": symbol,
                    "Open": round(last_row['Open'], 3),
                    "High": round(last_row['High'], 3),
                    "Low": round(last_row['Low'], 3),
                    "Close": round(last_row['Close'], 3),
                    "Volume": int(last_row['Volume'])
                })
            else:
                print(f"⚠️ Pas de données pour {name} (Marché fermé ?)")
                
        except Exception as e:
            print(f"❌ Erreur sur {name}: {e}")

    df_output = pd.DataFrame(results)
    return df_output
if __name__ == "__main__":
    monitor_df = get_market_monitoring()
    
    if not monitor_df.empty:
        print(monitor_df.to_string(index=False))
    else:
        print("Aucune donnée collectée.")