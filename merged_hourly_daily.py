import pandas as pd

df_h = pd.read_csv('wti_features_v3.csv')
df_d = pd.read_csv('daily_indicators.csv')

df_h['Date'] = pd.to_datetime(df_h['Datetime']).dt.strftime('%Y-%m-%d')
df_d['Date'] = pd.to_datetime(df_d['Date']).dt.strftime('%Y-%m-%d')

res = pd.merge(df_h, df_d, on='Date', how='left').drop(columns=['Date'])

res.to_csv('merged_data_final.csv', index=False)