import random
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse


# =========================
# CONFIGURATION
# =========================

# Prix (Yahoo Finance)
PRICE_SYMBOL = "CL=F"
PRICE_LOOKBACK_YEARS = 5
PRICE_OUTPUT_CSV = "crude_oil_wti_daily_last_5_years.csv"

END_DATE = datetime.today().strftime("%Y-%m-%d")

# News (GDELT) — requête ciblée sur prix / analyses / prévisions
QUERY = '(("oil price" OR "oil prices" OR "crude oil" OR WTI OR Brent OR OPEC) AND (price OR forecast OR outlook OR analysis OR opinion OR rally OR surge OR spike))'
NEWS_OUTPUT_JSON = "oil_news_precise_oilprice.json"
NEWS_OUTPUT_CSV = "oil_news_precise_oilprice.csv"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
NEWSAPI_KEY = "c0c8ec9b3bbf4d088c7cbaee6c77637e"
NEWSAPI_URL = "https://newsapi.org/v2/everything"
NEWSAPI_PAGE_SIZE = 100
NEWSAPI_MAX_PAGES = 20

# Reviews (Reddit only)
REVIEW_QUERY = "oil price OR crude oil OR WTI OR Brent OR OPEC"
REVIEWS_END_DATE = datetime.today()
REVIEWS_START_DATE = REVIEWS_END_DATE - timedelta(days=4 * 365)
REVIEWS_OUTPUT_CSV = "oil_reviews_last_4_years_sorted.csv"
REDDIT_SEARCH_URL = "https://www.reddit.com/search.json"
REDDIT_PAGE_SIZE = 100
MAX_REVIEW_POSTS = 2000

# Réduction du risque 429: moins d'appels + pause + retry/backoff
WINDOW_DAYS = 14
REQUEST_TIMEOUT = 30
MAX_RETRIES = 6
BASE_DELAY_SECONDS = 2.5
MAX_RECORDS = 100
NEWS_LOOKBACK_YEARS = 5


def years_ago(years):
    """Retourne la date correspondant à N années avant aujourd'hui."""
    today = datetime.today()
    try:
        return today.replace(year=today.year - years)
    except ValueError:
        return today.replace(month=2, day=28, year=today.year - years)


PRICE_START_DATE = years_ago(PRICE_LOOKBACK_YEARS).strftime("%Y-%m-%d")
START_DATE = years_ago(NEWS_LOOKBACK_YEARS).strftime("%Y-%m-%d")


def download_price_data():
    """Télécharge et sauvegarde les prix WTI journaliers."""
    # yfinance traite souvent la date de fin comme exclusive.
    end_for_yf = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    df_price = yf.download(
        PRICE_SYMBOL,
        start=PRICE_START_DATE,
        end=end_for_yf,
        interval="1d",
        progress=False,
    )

    if df_price.empty:
        print("Avertissement: aucun prix téléchargé.")
        return

    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = df_price.columns.get_level_values(0)

    df_price = df_price.reset_index()
    df_price.rename(columns={df_price.columns[0]: "Date"}, inplace=True)
    df_price["Date"] = pd.to_datetime(df_price["Date"]).dt.strftime("%Y-%m-%d")
    df_price = df_price.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    calendar_index = pd.date_range(df_price["Date"].min(), df_price["Date"].max(), freq="D")
    df_price["Date"] = pd.to_datetime(df_price["Date"])
    df_price = df_price.set_index("Date").reindex(calendar_index)
    df_price.index.name = "Date"

    if "Close" in df_price.columns:
        df_price["Close"] = df_price["Close"].ffill().bfill()
        for column in ["Open", "High", "Low"]:
            if column in df_price.columns:
                df_price[column] = df_price[column].fillna(df_price["Close"])

    if "Volume" in df_price.columns:
        df_price["Volume"] = df_price["Volume"].fillna(0)

    df_price = df_price.reset_index()
    df_price["Date"] = df_price["Date"].dt.strftime("%Y-%m-%d")

    df_price.to_csv(PRICE_OUTPUT_CSV, index=False)
    print(
        f"Prix sauvegardés: {PRICE_OUTPUT_CSV} "
        f"({len(df_price)} lignes continues, depuis {PRICE_START_DATE})"
    )


def build_session():
    """Construit une session HTTP avec retries auto sur erreurs temporaires."""
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "oil-news-collector/1.0"})
    return session


def format_gdelt_datetime(date_obj):
    """Convertit au format datetime GDELT: YYYYMMDDHHMMSS."""
    return date_obj.strftime("%Y%m%d%H%M%S")


def fetch_gdelt_news(session, start_datetime, end_datetime):
    """Récupère les articles GDELT pour une fenêtre, avec backoff manuel."""
    params = {
        "query": QUERY,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": format_gdelt_datetime(start_datetime),
        "enddatetime": format_gdelt_datetime(end_datetime),
        "maxrecords": MAX_RECORDS,
        "sort": "datedesc",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(GDELT_URL, params=params, timeout=REQUEST_TIMEOUT)

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError:
                    print(
                        f"Réponse non-JSON pour {start_datetime.date()} -> {end_datetime.date()}"
                    )
                    return []
                return payload.get("articles", [])

            if response.status_code == 429:
                wait_seconds = min(90, BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
                wait_seconds += random.uniform(0.3, 1.2)
                print(
                    f"429 reçu ({start_datetime.date()} -> {end_datetime.date()}) "
                    f"tentative {attempt}/{MAX_RETRIES}, pause {wait_seconds:.1f}s"
                )
                time.sleep(wait_seconds)
                continue

            print(
                f"Erreur HTTP {response.status_code} "
                f"({start_datetime.date()} -> {end_datetime.date()})"
            )
            return []

        except requests.RequestException as exc:
            wait_seconds = min(90, BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
            wait_seconds += random.uniform(0.3, 1.2)
            print(
                f"Erreur réseau ({start_datetime.date()} -> {end_datetime.date()}): {exc}. "
                f"Nouvelle tentative dans {wait_seconds:.1f}s"
            )
            time.sleep(wait_seconds)

    print(f"Fenêtre ignorée après échecs répétés: {start_datetime.date()} -> {end_datetime.date()}")
    return []


def clean_article(article):
    """Conserve uniquement les colonnes utiles pour l'analyse."""
    return {
        "date": article.get("seendate") or article.get("publishedAt"),
        "title": article.get("title"),
        "url": article.get("url"),
        "source": article.get("sourceCountry") or (article.get("source") if isinstance(article.get("source"), str) else article.get("source", {}).get("name") if article.get("source") else None),
        "language": article.get("language"),
        "domain": article.get("domain") or (urlparse(article.get("url")).netloc if article.get("url") else None),
    }


def fetch_newsapi_page(session, from_date, to_date, page=1):
    """Récupère une page d'articles depuis NewsAPI avec retries simples."""
    params = {
        "q": QUERY,
        "from": from_date.strftime("%Y-%m-%d"),
        "to": to_date.strftime("%Y-%m-%d"),
        "pageSize": NEWSAPI_PAGE_SIZE,
        "page": page,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWSAPI_KEY,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(NEWSAPI_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    print(f"NewsAPI: réponse non-JSON pour page {page}")
                    return None

            if resp.status_code == 429:
                wait_seconds = min(120, BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
                wait_seconds += random.uniform(0.3, 1.2)
                print(f"NewsAPI 429 tentative {attempt}/{MAX_RETRIES}, pause {wait_seconds:.1f}s")
                time.sleep(wait_seconds)
                continue

            print(f"NewsAPI HTTP {resp.status_code}: {resp.text}")
            return None

        except requests.RequestException as exc:
            wait_seconds = min(120, BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
            wait_seconds += random.uniform(0.3, 1.2)
            print(f"NewsAPI réseau: {exc}; retry dans {wait_seconds:.1f}s")
            time.sleep(wait_seconds)

    return None


def collect_news_newsapi():
    """Collecte les news via NewsAPI sur la période START_DATE -> END_DATE."""
    session = build_session()
    start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE, "%Y-%m-%d")
    all_articles = []

    print(f"Collecte NewsAPI de {START_DATE} à {END_DATE}...")

    page = 1
    while page <= NEWSAPI_MAX_PAGES:
        payload = fetch_newsapi_page(session, start_date, end_date, page=page)
        if not payload:
            break

        articles = payload.get("articles", [])
        if not articles:
            break

        for a in articles:
            mapped = {
                "publishedAt": a.get("publishedAt"),
                "title": a.get("title"),
                "url": a.get("url"),
                "source": a.get("source", {}).get("name") if a.get("source") else None,
                "language": a.get("language", None),
                "domain": urlparse(a.get("url")).netloc if a.get("url") else None,
            }
            all_articles.append(mapped)

        total_results = payload.get("totalResults", 0)
        fetched = page * NEWSAPI_PAGE_SIZE
        print(f"NewsAPI page {page} récupérée, articles: {len(articles)}, totalResults approx: {total_results}")
        if fetched >= total_results:
            break

        page += 1
        time.sleep(BASE_DELAY_SECONDS)

    return all_articles


def collect_news():
    """Collecte les news sur toute la période en fenêtres de plusieurs jours."""
    start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE, "%Y-%m-%d")
    final_boundary = end_date + timedelta(days=1)

    total_days = (final_boundary - start_date).days
    total_windows = (total_days + WINDOW_DAYS - 1) // WINDOW_DAYS

    all_articles = []
    current = start_date
    window_index = 0
    session = build_session()

    print(f"Collecte des news de {START_DATE} à {END_DATE}...")

    while current < final_boundary:
        next_date = min(current + timedelta(days=WINDOW_DAYS), final_boundary)
        window_index += 1

        print(
            f"Fenêtre {window_index}/{total_windows}: "
            f"{current.date()} -> {(next_date - timedelta(days=1)).date()}"
        )

        articles = fetch_gdelt_news(session, current, next_date)
        all_articles.extend(clean_article(a) for a in articles)

        # Pause minimale entre fenêtres pour éviter les bursts de requêtes.
        time.sleep(BASE_DELAY_SECONDS)
        current = next_date

    return all_articles


def save_news(articles):
    """Déduplique et sauvegarde les news en CSV/JSON."""
    df_news = pd.DataFrame(articles)

    if df_news.empty:
        print("Aucun article trouvé.")
        return

    df_news.dropna(subset=["title", "url"], inplace=True)
    df_news.drop_duplicates(subset=["url"], inplace=True)

    df_news.to_csv(NEWS_OUTPUT_CSV, index=False, encoding="utf-8")
    df_news.to_json(NEWS_OUTPUT_JSON, orient="records", indent=4, force_ascii=False)

    print("Collecte news terminée.")
    print("Nombre d'articles:", len(df_news))
    print("Fichier CSV:", NEWS_OUTPUT_CSV)
    print("Fichier JSON:", NEWS_OUTPUT_JSON)


def build_reddit_session():
    """Construit une session HTTP dédiée à Reddit."""
    session = build_session()
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 oil-reviews-scraper/1.0 (contact: local-script)"}
    )
    return session


def fetch_reddit_reviews_page(session, after_token=None):
    """Récupère une page Reddit avec gestion des erreurs 429 et réseau."""
    params = {
        "q": REVIEW_QUERY,
        "sort": "new",
        "t": "all",
        "type": "link",
        "limit": REDDIT_PAGE_SIZE,
        "raw_json": 1,
    }
    if after_token:
        params["after"] = after_token

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(REDDIT_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT)

            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    print("Reddit: réponse non JSON")
                    return None

            if response.status_code == 429:
                wait_seconds = min(120, BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
                wait_seconds += random.uniform(0.3, 1.2)
                print(
                    f"Reddit 429 tentative {attempt}/{MAX_RETRIES}, pause {wait_seconds:.1f}s"
                )
                time.sleep(wait_seconds)
                continue

            print(f"Reddit HTTP {response.status_code}")
            return None

        except requests.RequestException as exc:
            wait_seconds = min(120, BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
            wait_seconds += random.uniform(0.3, 1.2)
            print(f"Reddit réseau: {exc}; retry dans {wait_seconds:.1f}s")
            time.sleep(wait_seconds)

    return None


def clean_review_post(post):
    """Normalise un post Reddit en enregistrement review."""
    created_utc = post.get("created_utc")
    created_dt = datetime.utcfromtimestamp(created_utc) if created_utc else None

    return {
        "date_utc": created_dt.strftime("%Y-%m-%d %H:%M:%S") if created_dt else None,
        "title": post.get("title"),
        "review_text": post.get("selftext"),
        "source": "reddit",
        "subreddit": post.get("subreddit"),
        "author": post.get("author"),
        "score": post.get("score"),
        "num_comments": post.get("num_comments"),
        "domain": "reddit.com",
        "url": f"https://www.reddit.com{post.get('permalink', '')}",
    }


def collect_reviews_last_4_years():
    """Collecte les reviews des 4 dernières années (Reddit seulement)."""
    session = build_reddit_session()
    after_token = None
    all_reviews = []

    start_dt = REVIEWS_START_DATE
    end_dt = REVIEWS_END_DATE

    print(
        "Collecte reviews Reddit (4 ans): "
        f"{start_dt.strftime('%Y-%m-%d')} -> {end_dt.strftime('%Y-%m-%d')}"
    )

    while len(all_reviews) < MAX_REVIEW_POSTS:
        payload = fetch_reddit_reviews_page(session, after_token=after_token)
        if not payload:
            break

        data = payload.get("data", {})
        children = data.get("children", [])
        after_token = data.get("after")

        if not children:
            break

        reached_older_range = False
        for child in children:
            post = child.get("data", {})
            created_utc = post.get("created_utc")
            if not created_utc:
                continue

            created_dt = datetime.utcfromtimestamp(created_utc)
            if created_dt < start_dt:
                reached_older_range = True
                continue
            if created_dt > end_dt:
                continue

            all_reviews.append(clean_review_post(post))
            if len(all_reviews) >= MAX_REVIEW_POSTS:
                break

        print(f"Reviews collectees: {len(all_reviews)}")

        if reached_older_range or not after_token:
            break

        time.sleep(BASE_DELAY_SECONDS)

    return all_reviews


def save_reviews(reviews):
    """Déduplique et sauvegarde les reviews en CSV."""
    df_reviews = pd.DataFrame(reviews)

    if df_reviews.empty:
        print("Aucune review trouvée.")
        return

    df_reviews.dropna(subset=["title", "url"], inplace=True)
    df_reviews.drop_duplicates(subset=["url"], inplace=True)

    df_reviews.to_csv(REVIEWS_OUTPUT_CSV, index=False, encoding="utf-8")

    print("Collecte reviews terminee.")
    print("Nombre de reviews:", len(df_reviews))
    print("Fichier CSV:", REVIEWS_OUTPUT_CSV)


def main():
    """Pipeline complet: prix + news + reviews 4 ans."""
    download_price_data()
    articles = collect_news()
    save_news(articles)
    reviews = collect_reviews_last_4_years()
    save_reviews(reviews)


if __name__ == "__main__":
    main()