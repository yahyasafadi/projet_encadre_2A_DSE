import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("⏳ Chargement des données...")

# Données 1 minute
df_1min = pd.read_csv('oil_intraday_1m.csv')
rename = {}
for col in df_1min.columns:
    if 'Datetime' in col or 'datetime' in col or 'timestamp' in col:
        rename[col] = 'Datetime'
df_1min = df_1min.rename(columns=rename)
df_1min['Datetime'] = pd.to_datetime(df_1min['Datetime'], utc=True)
df_1min = df_1min.sort_values('Datetime').reset_index(drop=True)
print(f"✅ Données 1min : {len(df_1min)} lignes")

# Données news horaires
df_news = pd.read_csv('news_wti_hourly_7d.csv')
df_news['timestamp'] = pd.to_datetime(df_news['timestamp'], utc=True, errors='coerce')
df_news = df_news.dropna(subset=['timestamp'])
df_news = df_news.sort_values('timestamp').reset_index(drop=True)
df_news['news_count'] = df_news['news_count'].fillna(0)
df_news['title']      = df_news['title'].fillna('no_news_available')
print(f"✅ Données news : {len(df_news)} lignes")

# ── MERGE ASOF ───────────────────────────────────────────────────
# Pour chaque minute, on prend la news de l'heure la plus récente
df_merged = pd.merge_asof(
    df_1min,
    df_news[['timestamp', 'title', 'news_count']],
    left_on  = 'Datetime',
    right_on = 'timestamp',
    direction = 'backward'   # prend la news précédente la plus proche
)

df_merged['news_count'] = df_merged['news_count'].fillna(0)
df_merged['title']      = df_merged['title'].fillna('no_news_available')
df_merged = df_merged.drop(columns=['timestamp'])

df_merged.to_csv('data_merged_1min.csv', index=False)
print(f"✅ {len(df_merged)} lignes mergées")
print(f"   Sauvegardé dans data_merged_1min.csv")
print(df_merged[['Datetime', 'title', 'news_count']].tail(3))