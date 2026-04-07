"""GDELT historical news loader for financial headline extraction.

GDELT (Global Database of Events, Language, and Tone) provides free access
to news articles worldwide.  We use the GDELT DOC 2.0 API for full-text
search of financial headlines covering the entire 2016-2026 backtest window.

Each ticker is cached individually so interrupted runs resume automatically.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

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


def _query_gdelt_doc(query, start_date, end_date, max_records=250):
    """Query GDELT DOC 2.0 API. Returns list of article dicts."""
    params = {
        "query": query,
        "mode": "artlist",
        "startdatetime": start_date,
        "enddatetime": end_date,
        "maxrecords": max_records,
        "format": "json",
        "sort": "datedesc",
    }
    for attempt in range(3):
        try:
            resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
            if resp.status_code == 200:
                text = resp.text.strip()
                if not text or text[0] != '{':
                    return []
                return resp.json().get("articles", [])
            elif resp.status_code == 429:
                time.sleep(2 * (2 ** attempt))
                continue
            else:
                return []
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return []


def _gdelt_date_fmt(dt):
    if isinstance(dt, str):
        dt = pd.Timestamp(dt)
    return dt.strftime("%Y%m%d%H%M%S")


def _ticker_cache_path(ticker, start, end):
    safe = ticker.replace("/", "_").replace("\\", "_")
    return os.path.join(CACHE_DIR, f"gdelt_{safe}_{start}_{end}.csv")


def _fetch_ticker_headlines(ticker, query, start_dt, end_dt, max_per_ticker,
                            chunk_months, delay):
    """Fetch headlines for one ticker across time chunks."""
    records = []
    chunk_start = start_dt
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
            if not pub_date:
                continue
            try:
                pub_dt = pd.Timestamp(pub_date[:8])
            except Exception:
                continue

            records.append({
                "ticker": ticker,
                "date": pub_dt,
                "headline": title,
                "source": f"GDELT:{art.get('domain', '')}",
                "tone": float(art.get("tone", 0.0) or 0.0),
            })

        chunk_start = chunk_end
        time.sleep(delay)

    return records


def load_gdelt_headlines(tickers, start, end, max_per_ticker=250,
                         chunk_months=12, delay=1.0):
    """Load historical financial headlines from GDELT.

    Per-ticker caching: each ticker is saved individually so interrupted
    runs resume where they left off. Fully cached runs return instantly.

    Args:
        tickers: list of ticker symbols or dict {name: ticker}
        start: start date string (e.g. "2016-01-01")
        end: end date string (e.g. "2026-04-01")
        max_per_ticker: max headlines per ticker per chunk (GDELT caps at 250)
        chunk_months: months per API query chunk (12 = ~10 calls/ticker)
        delay: seconds between API calls (rate limiting)
    Returns:
        DataFrame with columns [ticker, date, headline, source, tone]
    """
    if isinstance(tickers, dict):
        tickers = list(tickers.values())

    os.makedirs(CACHE_DIR, exist_ok=True)

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    all_tickers = [t for t in tickers + ["MARKET"] if t in GDELT_QUERIES]
    total = len(all_tickers)

    # Count chunks for ETA
    n_chunks = 0
    cur = start_dt
    while cur < end_dt:
        cur = min(cur + pd.DateOffset(months=chunk_months), end_dt)
        n_chunks += 1
    est_per_ticker = n_chunks * (delay + 1.0)

    # Separate cached vs uncached
    cached_dfs = []
    to_fetch = []
    for ticker in all_tickers:
        cp = _ticker_cache_path(ticker, start, end)
        if os.path.exists(cp):
            cached_dfs.append(pd.read_csv(cp, parse_dates=["date"]))
        else:
            to_fetch.append(ticker)

    n_cached = total - len(to_fetch)
    n_cached_headlines = sum(len(d) for d in cached_dfs)

    if not to_fetch:
        df = pd.concat(cached_dfs, ignore_index=True) if cached_dfs else pd.DataFrame()
        if not df.empty:
            df = df.drop_duplicates(subset=["headline"]).sort_values("date").reset_index(drop=True)
        print(f"  GDELT: {len(df)} headlines (all {total} tickers cached)")
        return df

    # Progress header
    workers = min(8, len(to_fetch))
    eta_min = len(to_fetch) * est_per_ticker / 60 / workers
    print(f"  GDELT: {n_cached}/{total} tickers cached ({n_cached_headlines} headlines)")
    print(f"  Fetching {len(to_fetch)} tickers "
          f"({n_chunks} chunks each, {workers} parallel workers, ETA ~{eta_min:.0f} min)...")

    # Fetch uncached tickers in parallel
    t0 = time.time()
    fetched_dfs = []
    done_count = 0

    def _fetch_and_cache(ticker):
        records = _fetch_ticker_headlines(
            ticker, GDELT_QUERIES[ticker], start_dt, end_dt,
            max_per_ticker, chunk_months, delay,
        )
        ticker_df = pd.DataFrame(records)
        if not ticker_df.empty:
            ticker_df["date"] = pd.to_datetime(ticker_df["date"])
            ticker_df = ticker_df.drop_duplicates(subset=["headline"])
            ticker_df.to_csv(_ticker_cache_path(ticker, start, end), index=False)
        return ticker, ticker_df

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_and_cache, t): t for t in to_fetch}
        for future in as_completed(futures):
            done_count += 1
            ticker, ticker_df = future.result()
            if not ticker_df.empty:
                fetched_dfs.append(ticker_df)

            elapsed = time.time() - t0
            remaining = len(to_fetch) - done_count
            eta_sec = (elapsed / done_count) * remaining
            eta_str = f"{eta_sec / 60:.1f}m" if eta_sec >= 60 else f"{eta_sec:.0f}s"
            print(f"    [{n_cached + done_count}/{total}] {ticker}: "
                  f"{len(ticker_df)} headlines | {remaining} left (~{eta_str})")

    # Combine
    all_dfs = cached_dfs + fetched_dfs
    df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if not df.empty:
        df = df.drop_duplicates(subset=["headline"]).sort_values("date").reset_index(drop=True)

    print(f"  GDELT total: {len(df)} unique headlines "
          f"({time.time() - t0:.0f}s elapsed)")
    return df


def load_gdelt_tone_timeseries(query, start, end, resolution="day"):
    """Load GDELT tone/volume time series for a search query."""
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
    return pd.concat(frames.values(), axis=1).ffill().fillna(0)


def make_gdelt_sentiment_features(headlines_df, normalize=True):
    """Create aggregate GDELT sentiment features per day."""
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

    daily["gdelt_tone_5d"] = daily["gdelt_tone_mean"].rolling(5, min_periods=1).mean()
    daily["gdelt_attention_5d"] = daily["gdelt_article_count"].rolling(5, min_periods=1).mean()

    if normalize:
        rolling_mean = daily.rolling(252, min_periods=60).mean()
        rolling_std = daily.rolling(252, min_periods=60).std()
        daily = (daily - rolling_mean) / (rolling_std + 1e-8)
        daily = daily.clip(-3, 3).fillna(0)

    return daily
