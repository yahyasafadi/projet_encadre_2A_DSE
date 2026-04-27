import pandas as pd 

news_file = "news_wti_hourly_7d.csv"

oil_data_commodity="oil_intraday_1h.csv"

df_news_data=pd.read_csv(news_file)

df_commodity_data = pd.read_csv(oil_data_commodity)


df_news_data["timestamp"] = pd.to_datetime(df_news_data["timestamp"]).dt.tz_localize(None)
df_commodity_data["Datetime"] = pd.to_datetime(df_commodity_data["Datetime"]).dt.tz_localize(None)

df_final = pd.merge(
    df_commodity_data,
    df_news_data,
    left_on="Datetime",
    right_on="timestamp",
    how="left"
    
)

df_final["news_count"] = df_final["news_count"].fillna(0)

df_final["title"] = df_final["title"].fillna("no_news_available")


df_final.to_csv("data_merged.csv")

