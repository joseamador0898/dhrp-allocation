"""GDELT historical news loader for financial headline extraction.

GDELT (Global Database of Events, Language, and Tone) provides free access
to news articles worldwide.  We use the GDELT DOC 2.0 API for full-text
search of financial headlines covering the entire 2016-2026 backtest window.

This is the primary historical text source for LLM-DHRP, providing thousands
of time-stamped financial headlines that yfinance/RSS feeds cannot cover
(those return only recent articles).
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")

# GDELT DOC 2.0 API endpoint (free, no key required)
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Search queries per asset class / ticker
# Tested format: parenthesized groups with sourcelang:eng filter work reliably.
# Avoid bare double-quoted phrases at the top level (causes JSON parse errors).
GDELT_QUERIES = {
    # DM universe
    "SPY": '("SP500" OR "stock market" OR "Wall Street") sourcelang:eng',
    "QQQ": '("Nasdaq" OR "tech stocks" OR "technology sector") sourcelang:eng',
    "IWM": '("Russell 2000" OR "small cap") sourcelang:eng',
    "EFA": '("international stocks" OR "EAFE" OR "developed markets") sourcelang:eng',
    "VGK": '("European stocks" OR "Eurozone economy" OR "Stoxx") sourcelang:eng',
    "TLT": '("Treasury bond" OR "yield curve" OR "interest rate") sourcelang:eng',
    "IEF": '("Treasury note" OR "bond market" OR "fixed income") sourcelang:eng',
    "LQD": '("corporate bond" OR "investment grade" OR "credit market") sourcelang:eng',
    "VNQ": '("real estate" OR "REIT" OR "housing market") sourcelang:eng',
    "UUP": '("US dollar" OR "dollar index" OR "currency market") sourcelang:eng',
    # EM universe
    "EEM": '("emerging market" OR "developing economies") sourcelang:eng',
    "EWZ": '("Brazil economy" OR "Bovespa" OR "Brazilian real") sourcelang:eng',
    "FXI": '("China stocks" OR "Chinese economy" OR "Shanghai index") sourcelang:eng',
    "EWY": '("South Korea economy" OR "KOSPI" OR "Korean won") sourcelang:eng',
    "EWT": '("Taiwan stocks" OR "TSMC" OR "Taiwan economy") sourcelang:eng',
    "INDA": '("India stocks" OR "Sensex" OR "Nifty" OR "Indian economy") sourcelang:eng',
    "EWW": '("Mexico economy" OR "Mexican peso") sourcelang:eng',
    "EZA": '("South Africa economy" OR "Johannesburg stocks") sourcelang:eng',
    "THD": '("Thailand economy" OR "Thai stocks") sourcelang:eng',
    "TUR": '("Turkey economy" OR "Turkish lira") sourcelang:eng',
    # Commodities universe
    "USO": '("crude oil" OR "Brent" OR "WTI" OR "OPEC") sourcelang:eng',
    "UNG": '("natural gas price" OR "LNG market") sourcelang:eng',
    "GLD": '("gold price" OR "gold market" OR "bullion") sourcelang:eng',
    "SLV": '("silver price" OR "silver market") sourcelang:eng',
    "DBA": '("agriculture commodities" OR "crop prices") sourcelang:eng',
    "DBC": '("commodity index" OR "commodity market" OR "raw materials") sourcelang:eng',
    "CPER": '("copper price" OR "copper market" OR "industrial metals") sourcelang:eng',
    "WEAT": '("wheat price" OR "grain prices") sourcelang:eng',
    "CORN": '("corn price" OR "corn market") sourcelang:eng',
    "SOYB": '("soybean price" OR "soybean market") sourcelang:eng',
    # Market-wide
    "MARKET": '("financial markets" OR "stock market" OR "global economy") sourcelang:eng',
}

# Domains to prioritize (financial news sources)
FINANCIAL_DOMAINS = {
    "reuters.com", "bloomberg.com", "cnbc.com", "wsj.com", "ft.com",
    "marketwatch.com", "finance.yahoo.com", "investing.com", "seekingalpha.com",
    "barrons.com", "fool.com", "thestreet.com", "benzinga.com", "zacks.com",
    "kitco.com", "oilprice.com", "mining.com", "agweb.com",
}


def _query_gdelt_doc(query, start_date, end_date, max_records=250, mode="artlist",
                     retries=3, backoff=2.0):
    """Query the GDELT DOC 2.0 API for article metadata.

    Args:
        query: search query string
        start_date: YYYYMMDDHHMMSS format
        end_date: YYYYMMDDHHMMSS format
        max_records: maximum articles to return
        mode: "artlist" for article list, "timelinevol" for volume
        retries: number of retry attempts on 429/5xx
        backoff: base backoff delay in seconds
    Returns:
        list of dicts with article metadata
    """
    params = {
        "query": query,
        "mode": mode,
        "startdatetime": start_date,
        "enddatetime": end_date,
        "maxrecords": max_records,
        "format": "json",
        "sort": "datedesc",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
            if resp.status_code == 200:
                text = resp.text.strip()
                if not text or text[0] != '{':
                    return []
                data = resp.json()
                return data.get("articles", [])
            elif resp.status_code == 429:
                wait = backoff * (2 ** attempt)
                time.sleep(wait)
                continue
            else:
                return []
        except Exception:
            if attempt < retries - 1:
                time.sleep(backoff)
    return []


def _gdelt_date_fmt(dt):
    """Convert date to GDELT format YYYYMMDDHHMMSS."""
    if isinstance(dt, str):
        dt = pd.Timestamp(dt)
    return dt.strftime("%Y%m%d%H%M%S")


def load_gdelt_headlines(tickers, start, end, max_per_ticker=500,
                         chunk_months=6, delay=2.0):
    """Load historical financial headlines from GDELT for the full backtest period.

    Queries GDELT DOC 2.0 API in chunks to cover the entire date range.
    Results are cached to avoid redundant API calls.

    Args:
        tickers: list of ticker symbols or dict {name: ticker}
        start: start date string (e.g. "2016-01-01")
        end: end date string (e.g. "2026-04-01")
        max_per_ticker: max headlines per ticker per chunk
        chunk_months: months per API query chunk
        delay: seconds between API calls (rate limiting)
    Returns:
        DataFrame with columns [ticker, date, headline, source, tone]
    """
    if isinstance(tickers, dict):
        tickers = list(tickers.values())

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = f"gdelt_{start}_{end}_{'_'.join(sorted(tickers)[:5])}"
    cp = os.path.join(CACHE_DIR, f"{cache_key}.csv")
    if os.path.exists(cp):
        df = pd.read_csv(cp, parse_dates=["date"])
        print(f"  GDELT headlines: {len(df)} (cached)")
        return df

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    all_records = []
    total_tickers = len(tickers) + 1  # +1 for MARKET query

    for idx, ticker in enumerate(tickers + ["MARKET"]):
        query = GDELT_QUERIES.get(ticker)
        if not query:
            continue

        # Query in time chunks to get better coverage
        chunk_start = start_dt
        ticker_count = 0
        while chunk_start < end_dt:
            chunk_end = min(chunk_start + pd.DateOffset(months=chunk_months), end_dt)

            articles = _query_gdelt_doc(
                query=query,
                start_date=_gdelt_date_fmt(chunk_start),
                end_date=_gdelt_date_fmt(chunk_end),
                max_records=max_per_ticker,
            )

            for art in articles:
                title = art.get("title", "").strip()
                if not title or len(title) < 10:
                    continue
                pub_date = art.get("seendate", "")
                if pub_date:
                    try:
                        pub_dt = pd.Timestamp(pub_date[:8])
                    except Exception:
                        continue
                else:
                    continue

                tone = art.get("tone", 0.0)
                domain = art.get("domain", "")

                all_records.append({
                    "ticker": ticker,
                    "date": pub_dt,
                    "headline": title,
                    "source": f"GDELT:{domain}",
                    "tone": float(tone) if tone else 0.0,
                })
                ticker_count += 1

            chunk_start = chunk_end
            time.sleep(delay)

        if ticker_count > 0:
            print(f"    [{idx+1}/{total_tickers}] {ticker}: {ticker_count} headlines")

    df = pd.DataFrame(all_records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["headline"])
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(cp, index=False)

    print(f"  GDELT total: {len(df)} unique headlines")
    return df


def load_gdelt_tone_timeseries(query, start, end, resolution="day"):
    """Load GDELT tone/volume time series for a search query.

    Uses the timelinevol and timelinetone modes to get aggregate
    media attention and sentiment over time. Useful as an additional
    macro feature.

    Args:
        query: GDELT search query
        start: start date
        end: end date
        resolution: "day" or "month"
    Returns:
        DataFrame with columns [date, volume, tone]
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    frames = {}
    for mode in ["timelinevol", "timelinetone"]:
        params = {
            "query": query,
            "mode": mode,
            "startdatetime": _gdelt_date_fmt(start),
            "enddatetime": _gdelt_date_fmt(end),
            "format": "csv",
        }
        try:
            resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
            if resp.status_code == 200 and resp.text.strip():
                from io import StringIO
                ts = pd.read_csv(StringIO(resp.text), parse_dates=[0])
                if not ts.empty:
                    ts.columns = ["date", mode.replace("timeline", "")]
                    ts = ts.set_index("date")
                    frames[mode] = ts
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames.values(), axis=1).ffill().fillna(0)
    return combined


def make_gdelt_sentiment_features(headlines_df, normalize=True):
    """Create aggregate GDELT sentiment features per day.

    Aggregates headline tone into daily features: mean tone, tone volatility,
    article count (media attention), positive/negative ratio.

    Args:
        headlines_df: DataFrame from load_gdelt_headlines with 'tone' column
        normalize: whether to z-score normalize
    Returns:
        DataFrame with daily sentiment features
    """
    if headlines_df.empty or "tone" not in headlines_df.columns:
        return pd.DataFrame()

    df = headlines_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    daily = df.groupby(df["date"].dt.date).agg(
        gdelt_tone_mean=("tone", "mean"),
        gdelt_tone_std=("tone", "std"),
        gdelt_article_count=("tone", "count"),
        gdelt_pos_ratio=("tone", lambda x: (x > 0).mean()),
    ).fillna(0)

    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    # Smoothed features (5-day rolling)
    daily["gdelt_tone_5d"] = daily["gdelt_tone_mean"].rolling(5, min_periods=1).mean()
    daily["gdelt_attention_5d"] = daily["gdelt_article_count"].rolling(5, min_periods=1).mean()

    if normalize:
        rolling_mean = daily.rolling(252, min_periods=60).mean()
        rolling_std = daily.rolling(252, min_periods=60).std()
        daily = (daily - rolling_mean) / (rolling_std + 1e-8)
        daily = daily.clip(-3, 3).fillna(0)

    return daily
