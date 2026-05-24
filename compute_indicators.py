import pandas as pd
import numpy as np
import re
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing via EWM with alpha=1/period
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr


def load_price_data() -> pd.DataFrame:
    """Charge le CSV de prix, quel que soit son ancien ou nouveau format."""
    candidates = [
        Path('crude_oil_wti_daily_last_5_years.csv'),
        Path('crude_oil_wti_daily_last_4_years.csv'),
    ]

    infile = next((path for path in candidates if path.exists()), None)
    if infile is None:
        raise FileNotFoundError('Aucun fichier de prix WTI trouvé.')

    try:
        df = pd.read_csv(infile, parse_dates=['Date'])
        if 'Date' in df.columns:
            return df
    except ValueError:
        pass

    return pd.read_csv(
        infile,
        header=None,
        skiprows=3,
        names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'],
        parse_dates=['Date'],
    )


def reindex_to_calendar_days(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstitue une série quotidienne continue en remplissant les jours manquants."""
    df = df.sort_values('Date').drop_duplicates(subset=['Date']).reset_index(drop=True)
    df = df.set_index('Date')

    full_index = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(full_index)
    df.index.name = 'Date'

    if 'Close' in df.columns:
        df['Close'] = df['Close'].ffill().bfill()
        for column in ['Open', 'High', 'Low']:
            if column in df.columns:
                df[column] = df[column].fillna(df['Close'])

    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0)

    return df.reset_index()


def load_daily_news() -> pd.DataFrame:
    """Charge les news et les agrège par jour pour obtenir une correspondance journalière."""
    news_path = Path('oil_news_precise_oilprice.csv')
    if not news_path.exists():
        return pd.DataFrame(columns=['news_date', 'news_title', 'news_url', 'news_domain', 'news_source', 'news_language', 'news_article_count', 'sentiment_score'])

    df_news = pd.read_csv(news_path, sep=';')
    if df_news.empty:
        return pd.DataFrame(columns=['news_date', 'news_title', 'news_url', 'news_domain', 'news_source', 'news_language', 'news_article_count', 'sentiment_score'])

    df_news['news_datetime'] = pd.to_datetime(df_news['date'], format='%Y%m%dT%H%M%SZ', errors='coerce')
    df_news = df_news.dropna(subset=['news_datetime', 'title']).copy()

    df_news['news_day'] = df_news['news_datetime'].dt.normalize()
    df_news['title'] = df_news['title'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

    analyzer = SentimentIntensityAnalyzer()

    oil_keywords = {
        'oil', 'crude', 'wti', 'brent', 'gasoline', 'diesel', 'fuel', 'opec',
        'barrel', 'refinery', 'hormuz', 'iran', 'petrol', 'energy', 'shipment'
    }

    def normalize_title(title: str) -> str:
        text = str(title)
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove repeated separators that add noise to sentiment models.
        text = re.sub(r'\s*[|/\\]+\s*', ' ', text)
        return text

    def title_relevance_weight(title: str) -> float:
        tokens = set(re.findall(r"[a-zA-Z']+", title.lower()))
        return 1.0 if tokens & oil_keywords else 0.35

    grouped_rows = []
    for news_day, group in df_news.sort_values(['news_day', 'news_datetime']).groupby('news_day', sort=True):
        titles = [normalize_title(title) for title in group['title'].tolist() if title]

        # Score each title individually, then use relevance-weighted average.
        scores = []
        weights = []
        for title in titles:
            score = analyzer.polarity_scores(title)['compound']
            weight = title_relevance_weight(title)
            scores.append(score)
            weights.append(weight)

        if scores:
            sentiment_score = float(np.average(np.array(scores), weights=np.array(weights)))
        else:
            sentiment_score = 0.0

        # Keep only top relevant titles for readability.
        title_weight_pairs = list(zip(titles, weights))
        title_weight_pairs.sort(key=lambda x: x[1], reverse=True)
        combined_title = ' | '.join([t for t, _ in title_weight_pairs[:5]])

        first_row = group.iloc[0]
        grouped_rows.append(
            {
                'news_date': news_day,
                'news_title': combined_title,
                'news_url': first_row.get('url'),
                'news_domain': first_row.get('domain'),
                'news_source': first_row.get('source'),
                'news_language': first_row.get('language'),
                'news_article_count': len(group),
                'sentiment_score': round(sentiment_score, 4),
            }
        )

    return pd.DataFrame(grouped_rows).sort_values('news_date').reset_index(drop=True)


def attach_news_to_prices(df_prices: pd.DataFrame, df_news: pd.DataFrame) -> pd.DataFrame:
    """Associe à chaque jour de prix la news journalière la plus proche."""
    df_prices = df_prices.copy()
    df_prices['Date'] = pd.to_datetime(df_prices['Date']).dt.normalize()

    if df_news.empty:
        df_prices['news_date'] = pd.NaT
        df_prices['news_title'] = pd.NA
        df_prices['news_url'] = pd.NA
        df_prices['news_domain'] = pd.NA
        df_prices['news_source'] = pd.NA
        df_prices['news_language'] = pd.NA
        df_prices['news_article_count'] = pd.NA
        df_prices['sentiment_score'] = pd.NA
        return df_prices

    df_news = df_news.sort_values('news_date').reset_index(drop=True)
    df_prices = df_prices.sort_values('Date').reset_index(drop=True)

    merged = pd.merge_asof(
        df_prices,
        df_news,
        left_on='Date',
        right_on='news_date',
        # Use the most recent news on-or-before the price date and
        # avoid matching to news that are far away in time.
        direction='backward',
        tolerance=pd.Timedelta('3D'),
    )

    merged['news_date'] = merged['news_date'].dt.strftime('%Y-%m-%d')
    return merged


def main():
    out = 'crude_wti_features.csv'

    df = load_price_data()
    df = reindex_to_calendar_days(df)
    df_news = load_daily_news()

    # Map scraped-news sentiment to the price timeline using daily anchors.
    # Scores are interpolated between real news dates (text-derived anchors only).
    if not df_news.empty:
        df_news['news_date'] = pd.to_datetime(df_news['news_date'])
        anchors = df_news.sort_values('news_date').reset_index(drop=True)

        # Attach nearest news metadata for context columns.
        df = df.sort_values('Date').reset_index(drop=True)
        df = pd.merge_asof(
            df,
            anchors[['news_date', 'news_title', 'news_url', 'news_domain', 'news_source', 'news_language', 'news_article_count', 'sentiment_score']],
            left_on='Date',
            right_on='news_date',
            direction='nearest',
        )

        # Build a continuous timeline score from scraped-news anchors only.
        # Inside anchor range: linear interpolation.
        # Outside anchor range: exponential decay from edge anchor scores.
        anchor_days = anchors['news_date'].map(pd.Timestamp.toordinal).to_numpy(dtype=float)
        anchor_scores = anchors['sentiment_score'].to_numpy(dtype=float)
        date_days = df['Date'].map(pd.Timestamp.toordinal).to_numpy(dtype=float)

        interpolated = np.interp(date_days, anchor_days, anchor_scores)
        tau_days = 900.0

        # Use non-neutral edge anchors when available to avoid flat zero histories.
        nz_idx = np.where(np.abs(anchor_scores) > 1e-6)[0]
        if len(nz_idx) > 0:
            left_edge_day = anchor_days[nz_idx[0]]
            left_edge_score = anchor_scores[nz_idx[0]]
            right_edge_day = anchor_days[nz_idx[-1]]
            right_edge_score = anchor_scores[nz_idx[-1]]
        else:
            left_edge_day = anchor_days[0]
            left_edge_score = anchor_scores[0]
            right_edge_day = anchor_days[-1]
            right_edge_score = anchor_scores[-1]

        left_mask = date_days < anchor_days[0]
        right_mask = date_days > anchor_days[-1]

        if left_mask.any():
            left_dist = left_edge_day - date_days[left_mask]
            interpolated[left_mask] = left_edge_score * np.exp(-left_dist / tau_days)

        if right_mask.any():
            right_dist = date_days[right_mask] - right_edge_day
            interpolated[right_mask] = right_edge_score * np.exp(-right_dist / tau_days)

        df['sentiment_score'] = np.clip(interpolated, -1.0, 1.0).round(6)

    else:
        df = attach_news_to_prices(df, df_news)

    close = df['Close']

    # Moving averages
    sma20 = close.rolling(window=20, min_periods=1).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()

    df['EMA_ratio'] = close / ema20
    df['Close_SMA20_ratio'] = close / sma20

    # RSI 14
    df['RSI_14'] = compute_rsi(close, period=14)

    # MACD histogram (12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd - signal

    # ATR 14
    df['ATR_14'] = compute_atr(df['High'], df['Low'], close, period=14)

    # Range intraday
    df['Range_intraday'] = df['High'] - df['Low']

    # hour
    df['hour'] = df['Date'].dt.hour

    # Normalize sparse metadata from source files (e.g. empty source field).
    df['news_source'] = df['news_source'].fillna('unknown')
    df['news_language'] = df['news_language'].fillna('N/A')

    # Keep columns similar to wti_features_ready.csv as far as possible
    out_cols = ['Date', 'EMA_ratio', 'Close_SMA20_ratio', 'RSI_14', 'MACD_Hist',
                'ATR_14', 'Range_intraday', 'hour', 'news_date', 'news_title',
                'news_url', 'news_domain', 'news_source', 'news_language',
                'news_article_count', 'sentiment_score']

    df_out = df[out_cols].rename(columns={'Date': 'Datetime'})

    df_out.to_csv(out, index=False)
    print(f'Wrote indicators to {out}')


if __name__ == '__main__':
    main()
