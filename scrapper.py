import requests
import time
import csv
from datetime import datetime
import json

API_KEY = "e972789b5d134e199db989b6027bb4ca"
SYMBOL = "CL"
INTERVAL = "1min"
URL = f"https://api.twelvedata.com/time_series?symbol={SYMBOL}&interval={INTERVAL}&outputsize=1&apikey={API_KEY}"


url = "https://newsapi.org/v2/everything"
params = {
   "q": "oil price OR crude oil OR OPEC",
    "language": "en",
    "from": "2020-01-01",
    "to": "2026-12-31",
    "sortBy": "publishedAt",
    "pageSize": 100,
    "apiKey": "c0c8ec9b3bbf4d088c7cbaee6c77637e"
}

response = requests.get(url, params=params)
data = response.json()
with open("oil_news.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("Fichier JSON créé : oil_news.json")


def get_last_price():
    response = requests.get(URL)
    data = response.json()

    if "values" not in data:
        print("Erreur API :", data)
        return None

    return data["values"][0]

def save_to_csv(row, filename="oil_prices.csv"):
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["datetime"],
            row["open"],
            row["high"],
            row["low"],
            row["close"]
        ])

with open("oil_prices.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["datetime", "open", "high", "low", "close"])

last_saved_time = None

while True:
    now = datetime.now()

    if now.second == 0:
        row = get_last_price()
        if row and row["datetime"] != last_saved_time:
            save_to_csv(row)
            last_saved_time = row["datetime"]
            print("Enregistré :", row)
        time.sleep(1)
    else:
        time.sleep(0.5)