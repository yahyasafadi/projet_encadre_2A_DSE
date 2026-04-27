from pygooglenews import GoogleNews
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_oil_news_7d():
    gn = GoogleNews(lang='en', country='US')
    all_articles = []
    
    for i in range(7):
        target_date = datetime.now() - timedelta(days=i)
        date_str = target_date.strftime('%Y-%m-%d')
        next_day = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Extraction des news pour le : {date_str}...")
        
        search = gn.search('WTI crude oil', from_=date_str, to_=next_day)
        
        for entry in search['entries']:
            source_name = "Unknown"
            if hasattr(entry, 'source'):
                source_name = entry.source.get('title', entry.source.get('text', 'Unknown'))
            
            all_articles.append({
                'timestamp': entry.get('published', datetime.now()),
                'title': entry.get('title', 'No Title'),
                'source': source_name,
                'link': entry.get('link', '')
            })
    
        time.sleep(1)

    df = pd.DataFrame(all_articles)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.sort_values('timestamp')
    
    return df

df_news = fetch_oil_news_7d()

df_news.set_index('timestamp', inplace=True)

df_hourly = df_news.resample('1h').agg({
    'title': lambda x: ' | '.join(x) if not x.empty else "",
    'source': 'count'
}).rename(columns={'source': 'news_count'})

df = pd.read_csv("news_wti_hourly_7d.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df_all = pd.concat([df, df_hourly])
df_all = df_all[~df_all.index.duplicated(keep="last")]

df_all = df_all.sort_index()
df_all.to_csv("news_wti_hourly_7d.csv")