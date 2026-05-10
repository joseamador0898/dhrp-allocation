"""src.data (consolidated). Original layout: ['feature_engineering.py', 'fred_loader.py', 'gdelt_loader.py', 'gemini_sentiment.py', 'gsquant_loader.py', 'llm_features.py', 'price_loader.py', 'spgci_loader.py', 'text_loader.py', 'universe_config.py']"""

from __future__ import annotations

# ====================================================================
# Module: feature_engineering.py
# ====================================================================
import numpy as np
import pandas as pd

from src.models import hrp_allocation

# Default feature dimension — increased from 48 to accommodate volume features
DEFAULT_FDIM = 64


def make_features(rets, fdim=DEFAULT_FDIM, is_em=False, volume=None):
    """Build feature vector from trailing return data.

    Args:
        rets: DataFrame of daily returns (window x n_assets)
        fdim: output feature dimension
        is_em: emerging markets flag (shorter momentum window)
        volume: DataFrame of daily volume (same index as rets), or None
    Returns:
        np.ndarray of shape (fdim,)
    """
    n = len(rets.columns)
    feats = list(np.clip(rets.mean().values * 252, -2, 2))
    feats += list(np.clip(rets.std().values * np.sqrt(252), 0.01, 2))

    mw = 10 if is_em else 21
    if len(rets) >= mw * 2:
        recent = rets.iloc[-mw:].mean().values
        prior = rets.iloc[-mw * 2 : -mw].mean().values
        feats += list(np.clip((recent - prior) * 252, -1, 1))
    else:
        feats += [0.0] * n

    if len(rets) >= 63:
        short_vol = rets.iloc[-21:].std().values + 1e-8
        long_vol = rets.std().values + 1e-8
        feats += list(np.clip(short_vol / long_vol, 0.5, 2.0))
    else:
        feats += [1.0] * n

    corr = rets.corr().values
    avg_corr = corr[np.triu_indices_from(corr, k=1)].mean() if corr.shape[0] > 1 else 0
    feats.append(np.clip(avg_corr, -1, 1))

    # Volume-based aggregate features (4 features, universe-size-invariant)
    if volume is not None and not volume.empty:
        common_cols = [c for c in rets.columns if c in volume.columns]
        if common_cols and len(volume) >= 20:
            vol_data = volume[common_cols].reindex(rets.index).ffill().bfill().fillna(0)
            vol_ma = vol_data.rolling(20, min_periods=10).mean()
            last_vol = vol_data.iloc[-1].values
            last_ma = vol_ma.iloc[-1].values + 1.0
            vol_ratio = last_vol / last_ma
            vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)

            # F1: average abnormal volume (market-wide activity signal)
            feats.append(np.clip(float(np.mean(vol_ratio) - 1), -2, 2))
            # F2: volume dispersion (divergence in activity across assets)
            feats.append(np.clip(float(np.std(vol_ratio)), 0, 2))
            # F3: fraction of assets with high volume (> 1.5x normal)
            feats.append(np.clip(float(np.mean(vol_ratio > 1.5)), 0, 1))
            # F4: cross-sectional volume-return correlation
            last_ret = np.abs(rets.iloc[-1][common_cols].values)
            if np.std(last_ret) > 1e-10 and np.std(vol_ratio) > 1e-10:
                vr_corr = np.corrcoef(vol_ratio, last_ret)[0, 1]
                feats.append(np.clip(float(np.nan_to_num(vr_corr, nan=0)), -1, 1))
            else:
                feats.append(0.0)
        else:
            feats += [0.0, 0.0, 0.0, 0.0]

    out = np.zeros(fdim, dtype=np.float32)
    out[: min(len(feats), fdim)] = np.nan_to_num(
        np.array(feats[:fdim], dtype=np.float32), nan=0.0
    )
    return out


def build_dataset(prices, window=252, step=5, is_em=False, volume=None, fdim=DEFAULT_FDIM,
                   train_end=None, return_dates=False):
    """Build training dataset: features, covariances, forward returns, HRP weights.

    Args:
        prices: DataFrame of adjusted close prices
        window: lookback window in trading days
        step: step size between samples
        is_em: emerging markets flag
        volume: DataFrame of daily volume (same date index as prices), or None
        fdim: feature dimension
        train_end: if provided, only use data up to this date for training samples
        return_dates: if True, also return sample dates (rets.index[t] for each
                      sample) so callers can build time-varying auxiliary features
                      (e.g. per-timestep text embeddings) aligned with the dataset
    Returns:
        (X, S, R, H) tuple, or (X, S, R, H, dates) if return_dates=True
    """
    rets = prices.pct_change().dropna()
    X, S, R, H, D = [], [], [], [], []
    fwd = 3 if is_em else 5

    # Limit training samples to avoid look-ahead bias
    max_t = len(rets) - fwd
    if train_end is not None:
        end_date = pd.Timestamp(train_end)
        end_candidates = rets.index[rets.index <= end_date]
        if len(end_candidates) > 0:
            max_t = min(max_t, rets.index.get_loc(end_candidates[-1]) + 1)

    for t in range(window, max_t, step):
        w_rets = rets.iloc[t - window : t]
        cov = w_rets.cov().values * 252
        if np.isnan(cov).any() or np.isinf(cov).any():
            continue
        if is_em:
            cov += np.eye(cov.shape[0]) * 0.01
        try:
            hrp_w = hrp_allocation(cov)
        except Exception:
            hrp_w = np.ones(cov.shape[0]) / cov.shape[0]
        if np.isnan(hrp_w).any():
            hrp_w = np.ones(cov.shape[0]) / cov.shape[0]

        vol_window = None
        if volume is not None and not volume.empty:
            vol_window = volume.iloc[max(0, t - window) : t]

        feat = make_features(w_rets, fdim, is_em, volume=vol_window)
        fwd_r = (1 + rets.iloc[t : t + fwd]).prod(axis=0).values - 1
        if np.isnan(feat).any() or np.isnan(fwd_r).any():
            continue
        X.append(feat)
        S.append(cov.astype(np.float32))
        R.append(fwd_r.astype(np.float32))
        H.append(hrp_w.astype(np.float32))
        D.append(rets.index[t - 1])  # last date in the feature window

    if X:
        result = (np.stack(X), np.stack(S), np.stack(R), np.stack(H))
        if return_dates:
            return result + (pd.DatetimeIndex(D),)
        return result
    empty = (np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,)))
    if return_dates:
        return empty + (pd.DatetimeIndex([]),)
    return empty

# ====================================================================
# Module: fred_loader.py
# ====================================================================
"""FRED macro feature extraction for DHRP inputs.

Provides macro-economic regime signals: VIX, yield curve, credit spreads,
fed funds rate, consumer sentiment, unemployment, CPI, and trade-weighted dollar.
These complement LLM features by capturing institutional market structure that
language models cannot directly observe.
"""

import os

import numpy as np
import pandas as pd


FRED_SERIES = {
    "vix": "VIXCLS",           # VIX index
    "yield_curve": "T10Y2Y",   # 10Y-2Y yield spread
    "credit_spread": "BAMLH0A0HYM2",  # High yield OAS
    "fed_rate": "FEDFUNDS",    # Fed funds rate
}

# Extended series for richer macro conditioning
FRED_SERIES_EXTENDED = {
    **FRED_SERIES,
    "consumer_sent": "UMCSENT",      # U of Michigan Consumer Sentiment
    "unemployment": "UNRATE",        # Unemployment rate
    "cpi": "CPIAUCSL",              # CPI (all urban consumers)
    "dollar_index": "DTWEXBGS",      # Trade-Weighted Dollar Index
}

# Full series — tested and verified accessible (37 series)
FRED_SERIES_FULL = {
    **FRED_SERIES_EXTENDED,
    # Treasury yields (daily)
    "dgs10": "DGS10",               # 10-Year Treasury yield
    "dgs2": "DGS2",                 # 2-Year Treasury yield
    "dgs30": "DGS30",               # 30-Year Treasury yield
    # Inflation expectations
    "breakeven_10y": "T10YIE",       # 10Y breakeven inflation
    "tips_real_10y": "DFII10",       # 10Y TIPS real yield
    # Fed balance sheet & liquidity
    "fed_balance": "WALCL",          # Fed total assets (weekly)
    "reverse_repo": "RRPONTSYD",     # Overnight reverse repo volume (daily)
    # FX rates (daily)
    "eurusd": "DEXUSEU",            # EUR/USD
    "usdjpy": "DEXJPUS",            # USD/JPY
    "gbpusd": "DEXUSUK",            # GBP/USD
    # Equity indices (daily)
    "nasdaq": "NASDAQCOM",           # Nasdaq Composite
    "sp500": "SP500",                # S&P 500
    # Financial conditions (weekly/daily)
    "nfci": "NFCI",                  # Chicago Fed National Financial Conditions
    # Labor market (weekly/monthly)
    "jobless_claims": "ICSA",        # Initial jobless claims (weekly)
    "nonfarm_payroll": "PAYEMS",     # Nonfarm payrolls (monthly)
    # Real economy (monthly)
    "industrial_prod": "INDPRO",     # Industrial production index
    "retail_sales": "RSAFS",         # Retail sales
    "housing_starts": "HOUST",       # Housing starts
    # Prices (monthly)
    "ppi": "PPIACO",                 # PPI all commodities
    "core_pce": "PCEPILFE",          # Core PCE price index
    # Money supply
    "m2": "M2SL",                    # M2 money supply (monthly)
    "monetary_base": "BOGMBASE",     # Monetary base (biweekly)
}

# Commodity-specific FRED price series (for Commodities universe)
FRED_SERIES_COMMODITY = {
    "wti_crude": "DCOILWTICO",       # WTI Crude Oil (daily)
    "brent_crude": "DCOILBRENTEU",   # Brent Crude Oil (daily)
    "gold_am_fix": "GOLDAMGBD228NLBM",  # Gold AM London Fix (daily)
    "copper": "PCOPPUSDM",           # Global Copper Price (monthly)
    "wheat_global": "PWHEAMTUSDM",   # Global Wheat Price (monthly)
    "corn_maize": "PMAIZMTUSDM",     # Global Corn/Maize Price (monthly)
}

# EM-specific FRED series (for Emerging Markets universe)
FRED_SERIES_EM = {
    "em_corp_oas": "BAMLEMCBPIOAS",       # EM Corporate Bond OAS (daily)
    "em_hy_oas": "BAMLEMHBHYCRPIOAS",     # EM High Yield Bond OAS (daily)
}


def load_fred_data(start, end, series=None, api_key=None, extended=True):
    """Load macro data from FRED API.

    Args:
        start: start date string
        end: end date string
        series: dict of {name: FRED_ID} or None for defaults
        api_key: FRED API key or None (reads from env)
        extended: if True and series is None, use extended series set
    Returns:
        DataFrame with macro features, forward-filled
    """
    from fredapi import Fred

    key = api_key or os.environ.get("FRED_API_KEY")
    if not key or key == "ROTATE_ME":
        print("Warning: FRED_API_KEY not set. Returning empty macro features.")
        return pd.DataFrame()

    fred = Fred(api_key=key)
    if series is None:
        series = FRED_SERIES_EXTENDED if extended else FRED_SERIES

    frames = {}
    for name, fred_id in series.items():
        try:
            s = fred.get_series(fred_id, observation_start=start, observation_end=end)
            frames[name] = s
        except Exception as e:
            print(f"Warning: Failed to load {name} ({fred_id}): {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().ffill().bfill()
    return df


def make_macro_features(fred_df, normalize=True):
    """Convert FRED data into a feature vector per date.

    Args:
        fred_df: DataFrame from load_fred_data
        normalize: whether to z-score normalize features
    Returns:
        DataFrame with normalized macro features
    """
    if fred_df.empty:
        return fred_df

    features = fred_df.copy()

    # Derived features from base series
    if "yield_curve" in features.columns:
        features["yc_change_5d"] = features["yield_curve"].diff(5)
    if "vix" in features.columns:
        features["vix_change_5d"] = features["vix"].pct_change(5)
        features["vix_above_20"] = (features["vix"] > 20).astype(float)
    if "credit_spread" in features.columns:
        features["cs_change_5d"] = features["credit_spread"].diff(5)

    # New derived features from extended series
    if "consumer_sent" in features.columns:
        features["sent_change_5d"] = features["consumer_sent"].pct_change(5)
    if "cpi" in features.columns:
        features["cpi_mom"] = features["cpi"].pct_change(1)  # month-over-month
    if "dollar_index" in features.columns:
        features["dollar_mom_21d"] = features["dollar_index"].pct_change(21)

    features = features.ffill().bfill().fillna(0)

    if normalize:
        # Rolling z-score (252-day window) to avoid look-ahead bias
        rolling_mean = features.rolling(252, min_periods=60).mean()
        rolling_std = features.rolling(252, min_periods=60).std()
        features = (features - rolling_mean) / (rolling_std + 1e-8)
        features = features.clip(-3, 3).fillna(0)

    return features


def get_macro_vector(macro_df, date, feature_names=None):
    """Get macro feature vector for a specific date.

    Args:
        macro_df: normalized macro features DataFrame
        date: target date
        feature_names: list of column names to use, or None for all
    Returns:
        np.ndarray of shape (n_features,)
    """
    if macro_df.empty:
        n = len(feature_names) if feature_names else 4
        return np.zeros(n, dtype=np.float32)

    date = pd.Timestamp(date)
    if feature_names:
        cols = [c for c in feature_names if c in macro_df.columns]
    else:
        cols = macro_df.columns.tolist()

    # Find closest available date
    idx = macro_df.index.get_indexer([date], method="ffill")[0]
    if idx < 0:
        return np.zeros(len(cols), dtype=np.float32)

    return macro_df.iloc[idx][cols].values.astype(np.float32)


def make_commodity_features(fred_df, normalize=True):
    """Convert commodity FRED data into regime features.

    Derived features: crude term spread, gold momentum, copper momentum,
    agriculture momentum, crude volatility regime.
    """
    if fred_df.empty:
        return fred_df

    features = fred_df.copy()

    # Crude oil term spread (Brent - WTI)
    if "brent_crude" in features.columns and "wti_crude" in features.columns:
        features["crude_term_spread"] = features["brent_crude"] - features["wti_crude"]

    # Momentum signals (21-day % change)
    for col in ["wti_crude", "gold_am_fix", "copper", "wheat_global", "corn_maize"]:
        if col in features.columns:
            features[f"{col}_mom_21d"] = features[col].pct_change(21)

    # Crude volatility regime (21-day rolling std of daily returns)
    if "wti_crude" in features.columns:
        crude_ret = features["wti_crude"].pct_change()
        features["crude_vol_21d"] = crude_ret.rolling(21, min_periods=10).std()

    features = features.ffill().bfill().fillna(0)

    if normalize:
        rolling_mean = features.rolling(252, min_periods=60).mean()
        rolling_std = features.rolling(252, min_periods=60).std()
        features = (features - rolling_mean) / (rolling_std + 1e-8)
        features = features.clip(-3, 3).fillna(0)

    return features


def make_em_features(fred_df, normalize=True):
    """Convert EM FRED data into credit regime features.

    Derived features: OAS level changes, credit stress indicator,
    spread widening/tightening momentum.
    """
    if fred_df.empty:
        return fred_df

    features = fred_df.copy()

    # OAS momentum (5-day changes)
    for col in ["em_corp_oas", "em_hy_oas"]:
        if col in features.columns:
            features[f"{col}_change_5d"] = features[col].diff(5)
            features[f"{col}_change_21d"] = features[col].diff(21)

    # Credit stress indicator: HY-IG spread differential
    if "em_hy_oas" in features.columns and "em_corp_oas" in features.columns:
        features["em_hy_ig_diff"] = features["em_hy_oas"] - features["em_corp_oas"]

    features = features.ffill().bfill().fillna(0)

    if normalize:
        rolling_mean = features.rolling(252, min_periods=60).mean()
        rolling_std = features.rolling(252, min_periods=60).std()
        features = (features - rolling_mean) / (rolling_std + 1e-8)
        features = features.clip(-3, 3).fillna(0)

    return features

# ====================================================================
# Module: gdelt_loader.py
# ====================================================================
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

# ====================================================================
# Module: gemini_sentiment.py
# ====================================================================
"""Gemini-powered structured financial sentiment extraction.

Replaces FinBERT + Qwen3-8B with Gemini API calls:
- 6-dimensional structured output per headline (vs FinBERT's 3-class scalar)
- Schema-enforced JSON (vs Qwen3's free-text parsing)
- Zero GPU compute (API calls)
- Works with any Gemini model (3.1 Pro, 3.1 Flash-Lite, 2.5 Flash, etc.)

Usage:
    from src.data.gemini_sentiment import extract_sentiment_batch
    features = extract_sentiment_batch(headlines_df, api_key=..., model='gemini-3.1-pro-preview')
    # features: dict[ticker, np.ndarray of shape (6,)]
"""

import json
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel


class HeadlineSentiment(BaseModel):
    """Structured sentiment label for a single financial headline."""
    sentiment_score: float       # -1.0 (very bearish) to 1.0 (very bullish)
    risk_level: str              # "low", "medium", "high", "extreme"
    regime: str                  # "risk_on", "risk_off", "neutral", "crisis"
    confidence: float            # 0.0 to 1.0
    sector_impact: str           # "broad_market", "sector_specific", "macro", "idiosyncratic"
    rate_sensitivity: float      # -1.0 (benefits from rate cuts) to 1.0 (benefits from rate hikes)


# Rebuild for Pydantic + Python 3.14 compatibility
HeadlineSentiment.model_rebuild()


# Encode categorical fields as floats for the feature vector
RISK_LEVEL_MAP = {"low": 0.0, "medium": 0.33, "high": 0.67, "extreme": 1.0}
REGIME_MAP = {"risk_on": 1.0, "neutral": 0.0, "risk_off": -0.5, "crisis": -1.0}
SECTOR_MAP = {"idiosyncratic": 0.0, "sector_specific": 0.33, "macro": 0.67, "broad_market": 1.0}


def sentiment_to_vector(s: HeadlineSentiment) -> np.ndarray:
    """Convert a HeadlineSentiment to a 6-dimensional float vector."""
    return np.array([
        s.sentiment_score,
        RISK_LEVEL_MAP.get(s.risk_level, 0.5),
        REGIME_MAP.get(s.regime, 0.0),
        s.confidence,
        SECTOR_MAP.get(s.sector_impact, 0.5),
        s.rate_sensitivity,
    ], dtype=np.float32)


GEMINI_FEATURE_DIM = 6  # number of features per headline


def _build_prompt(headlines: list[str]) -> str:
    """Build the prompt for batch headline analysis."""
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    return (
        "You are a quantitative financial analyst. Analyze the financial sentiment "
        "of each headline below for portfolio allocation purposes. For each headline, "
        "assess: (1) overall sentiment score from -1.0 (very bearish) to 1.0 (very "
        "bullish), (2) risk level, (3) market regime implied, (4) your confidence in "
        "the assessment, (5) whether the impact is broad-market, sector-specific, "
        "macro, or idiosyncratic, and (6) rate sensitivity from -1.0 (benefits from "
        "rate cuts) to 1.0 (benefits from rate hikes).\n\n"
        f"Headlines:\n{numbered}"
    )


def _call_gemini(
    client,
    headlines: list[str],
    model: str = "gemini-3.1-pro-preview",
    max_retries: int = 3,
) -> list[HeadlineSentiment]:
    """Call Gemini API with structured output for a batch of headlines."""
    from google import genai

    prompt = _build_prompt(headlines)

    # Raw JSON schema (avoids Pydantic + Python 3.14 Literal incompatibility
    # with the Gemini SDK's schema transformer)
    raw_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "sentiment_score": {"type": "number"},
                "risk_level": {"type": "string", "enum": ["low", "medium", "high", "extreme"]},
                "regime": {"type": "string", "enum": ["risk_on", "risk_off", "neutral", "crisis"]},
                "confidence": {"type": "number"},
                "sector_impact": {"type": "string", "enum": ["broad_market", "sector_specific", "macro", "idiosyncratic"]},
                "rate_sensitivity": {"type": "number"},
            },
            "required": ["sentiment_score", "risk_level", "regime", "confidence", "sector_impact", "rate_sensitivity"],
        },
    }

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=raw_schema,
                    temperature=0.1,
                ),
            )
            results = json.loads(response.text)
            return [HeadlineSentiment(**r) for r in results]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Gemini error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Gemini FAILED after {max_retries} attempts: {e}")
                # Return neutral defaults
                return [HeadlineSentiment(
                    sentiment_score=0.0, risk_level="medium", regime="neutral",
                    confidence=0.0, sector_impact="broad_market", rate_sensitivity=0.0,
                ) for _ in headlines]


def extract_sentiment_batch(
    headlines_df,
    api_key: str | None = None,
    model: str = "gemini-3.1-pro-preview",
    max_headlines_per_ticker: int = 15,
    batch_size: int = 10,
    cache_path: str | None = None,
    delay_between_calls: float = 1.0,
) -> dict[str, np.ndarray]:
    """Extract structured sentiment features for all tickers using Gemini API.

    Args:
        headlines_df: DataFrame with columns ['ticker', 'headline']
        api_key: Gemini API key (reads from GOOGLE_API_KEY env var if None)
        model: Gemini model name
        max_headlines_per_ticker: max headlines to analyze per ticker
        batch_size: headlines per API call (Gemini handles multi-headline prompts)
        cache_path: path to cache JSON (avoids re-calling API on re-runs)
        delay_between_calls: seconds between API calls (rate limiting)

    Returns:
        dict mapping ticker -> np.ndarray of shape (GEMINI_FEATURE_DIM,)
        Each vector is the MEAN of per-headline sentiment vectors for that ticker.
    """
    from google import genai

    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("No Gemini API key. Set GOOGLE_API_KEY or pass api_key=")

    # Load cache if exists
    cache_file = Path(cache_path) if cache_path else Path("results/features/gemini_sentiment.json")
    cached = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"  Loaded {len(cached)} cached ticker sentiments from {cache_file}")
        except Exception:
            cached = {}

    client = genai.Client(api_key=api_key)

    # Group headlines by ticker
    ticker_headlines = (
        headlines_df.groupby("ticker")["headline"]
        .apply(lambda x: list(x)[:max_headlines_per_ticker])
        .to_dict()
    )

    results = {}
    n_api_calls = 0
    n_cached = 0

    for ticker, hdls in sorted(ticker_headlines.items()):
        if len(hdls) < 3:
            continue

        # Check cache
        if ticker in cached:
            results[ticker] = np.array(cached[ticker], dtype=np.float32)
            n_cached += 1
            continue

        # Call Gemini in batches
        all_vectors = []
        for i in range(0, len(hdls), batch_size):
            batch = hdls[i:i + batch_size]
            sentiments = _call_gemini(client, batch, model=model)
            for s in sentiments:
                all_vectors.append(sentiment_to_vector(s))
            n_api_calls += 1
            if delay_between_calls > 0:
                time.sleep(delay_between_calls)

        if all_vectors:
            mean_vec = np.mean(all_vectors, axis=0).astype(np.float32)
            results[ticker] = mean_vec
            cached[ticker] = mean_vec.tolist()
            print(f"  {ticker}: {len(all_vectors)} headlines -> "
                  f"sentiment={mean_vec[0]:.2f}, risk={mean_vec[1]:.2f}, "
                  f"regime={mean_vec[2]:.2f}, rate_sens={mean_vec[5]:.2f}")

    # Save cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cached, f, indent=2)
    print(f"  Cached {len(cached)} tickers to {cache_file}")
    print(f"  API calls: {n_api_calls}, cached: {n_cached}")

    return results


def build_gemini_text_tensor(
    sentiment_by_ticker: dict[str, np.ndarray],
    n_samples: int,
    tickers: list[str],
) -> np.ndarray:
    """Build a (n_samples, GEMINI_FEATURE_DIM) tensor from per-ticker sentiment.

    The sentiment is STATIC per ticker (same value across all time steps),
    but the DHRP gating network can learn to route based on the cross-sectional
    dispersion of sentiment across assets at each timestep.

    For time-varying sentiment, call extract_sentiment_batch with date-windowed
    headlines.

    Args:
        sentiment_by_ticker: dict from extract_sentiment_batch()
        n_samples: number of time steps in the training dataset
        tickers: ordered list of ticker symbols matching the price DataFrame columns

    Returns:
        np.ndarray of shape (n_samples, GEMINI_FEATURE_DIM)
        The per-ticker vectors are averaged across tickers to produce a
        global sentiment signal per timestep (consistent with the existing
        LLM-DHRP pipeline that averages FinBERT embeddings across assets).
    """
    # Average across all tickers to get a single global sentiment vector
    vecs = []
    for t in tickers:
        if t in sentiment_by_ticker:
            vecs.append(sentiment_by_ticker[t])
        else:
            vecs.append(np.zeros(GEMINI_FEATURE_DIM, dtype=np.float32))

    global_vec = np.mean(vecs, axis=0).astype(np.float32)
    # Tile across all samples (static sentiment — same for all timesteps)
    return np.tile(global_vec, (n_samples, 1))

# ====================================================================
# Module: gsquant_loader.py
# ====================================================================
"""Goldman Sachs Quant (gs_quant) data integration.

Pulls daily close prices for macro indices (SPX, VIX, MXEF, BCOMTR, DXY, etc.)
and FX spot rates from GS Marquee via the TREOD and FXSPOT_STANDARD datasets.
Requires GS_CLIENT_ID and GS_CLIENT_SECRET in .env.
"""

import os

import numpy as np
import pandas as pd


CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")

# Index tickers to resolve via SecurityMaster (TREOD dataset)
# All verified accessible with read_product_data + read_financial_data scopes
GS_INDEX_TICKERS = {
    "spx": "SPX",          # S&P 500
    "vix": "VIX",          # CBOE VIX
    "ndx": "NDX",          # Nasdaq 100
    "rty": "RTY",          # Russell 2000
    "mxef": "MXEF",        # MSCI Emerging Markets
    "mxwo": "MXWO",        # MSCI World
    "mxea": "MXEA",        # MSCI EAFE
    "bcomtr": "BCOMTR",    # Bloomberg Commodity TR
    "dxy": "DXY",          # Dollar Index
    "stoxx": "SX5E",       # Euro Stoxx 50
    "nky": "NKY",          # Nikkei 225
    "hsi": "HSI",          # Hang Seng Index
    "shcomp": "SHCOMP",    # Shanghai Composite
}

# FX pairs to resolve via Bloomberg ID (FXSPOT_STANDARD dataset)
GS_FX_PAIRS = {
    "eurusd": "EURUSD",
    "usdjpy": "USDJPY",
}


def _get_session():
    """Authenticate with GS Quant using credentials from .env."""
    try:
        from gs_quant.session import GsSession
    except ImportError:
        print("Install gs_quant: pip install gs-quant")
        return None

    client_id = os.environ.get("GS_CLIENT_ID")
    client_secret = os.environ.get("GS_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("Warning: GS_CLIENT_ID/GS_CLIENT_SECRET not set.")
        return None

    try:
        GsSession.use(
            client_id=client_id,
            client_secret=client_secret,
            scopes=("read_product_data", "read_financial_data"),
        )
        return GsSession.current
    except Exception as e:
        print(f"Warning: GS Quant auth failed: {e}")
        return None


def _resolve_assets(tickers, id_type="ticker"):
    """Resolve ticker symbols to Marquee asset IDs."""
    from gs_quant.markets.securities import SecurityMaster, AssetIdentifier

    identifier = (
        AssetIdentifier.TICKER if id_type == "ticker"
        else AssetIdentifier.BLOOMBERG_ID
    )

    resolved = {}
    for name, symbol in tickers.items():
        try:
            asset = SecurityMaster.get_asset(symbol, identifier)
            if asset:
                resolved[name] = asset.get_marquee_id()
        except Exception:
            pass
    return resolved


def load_gs_index_data(start, end):
    """Load daily close prices for macro indices via TREOD dataset.

    Returns DataFrame with date index, one column per index (close prices).
    Covers: SPX, VIX, NDX, RTY, MXEF, MXWO, MXEA, BCOMTR, DXY.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"gs_indices_{start}_{end}.csv")
    if os.path.exists(cp):
        df = pd.read_csv(cp, index_col=0, parse_dates=True)
        print(f"  GS indices: {df.shape} (cached)")
        return df

    session = _get_session()
    if session is None:
        return pd.DataFrame()

    from gs_quant.api.gs.data import GsDataApi

    asset_ids = _resolve_assets(GS_INDEX_TICKERS, id_type="ticker")
    if not asset_ids:
        print("  No GS index assets resolved.")
        return pd.DataFrame()

    print(f"  Resolved {len(asset_ids)}/{len(GS_INDEX_TICKERS)} indices")

    frames = {}
    for name, mid in asset_ids.items():
        try:
            rows = GsDataApi.query_data(
                query={
                    "where": {"assetId": [mid]},
                    "startDate": str(start),
                    "endDate": str(end),
                },
                dataset_id="TREOD",
            )
            if rows:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"])
                series = df.set_index("date")["closePrice"].sort_index()
                series.name = name
                frames[name] = series
                print(f"    {name}: {len(series)} days")
        except Exception as e:
            print(f"    {name}: {str(e)[:80]}")

    if not frames:
        return pd.DataFrame()

    combined = pd.DataFrame(frames).sort_index().ffill().bfill()
    combined.to_csv(cp)
    print(f"  GS indices: {combined.shape}")
    return combined


def load_gs_fx_data(start, end):
    """Load daily FX spot rates via FXSPOT_STANDARD dataset.

    Returns DataFrame with date index, one column per pair.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"gs_fx_{start}_{end}.csv")
    if os.path.exists(cp):
        df = pd.read_csv(cp, index_col=0, parse_dates=True)
        print(f"  GS FX: {df.shape} (cached)")
        return df

    session = _get_session()
    if session is None:
        return pd.DataFrame()

    from gs_quant.api.gs.data import GsDataApi

    asset_ids = _resolve_assets(GS_FX_PAIRS, id_type="bbg")
    if not asset_ids:
        print("  No GS FX assets resolved.")
        return pd.DataFrame()

    frames = {}
    for name, mid in asset_ids.items():
        try:
            rows = GsDataApi.query_data(
                query={
                    "where": {"assetId": [mid]},
                    "startDate": str(start),
                    "endDate": str(end),
                },
                dataset_id="FXSPOT_STANDARD",
            )
            if rows:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"])
                series = df.set_index("date")["spot"].sort_index()
                series.name = name
                frames[name] = series
                print(f"    {name}: {len(series)} days")
        except Exception as e:
            print(f"    {name}: {str(e)[:80]}")

    if not frames:
        return pd.DataFrame()

    combined = pd.DataFrame(frames).sort_index().ffill().bfill()
    combined.to_csv(cp)
    print(f"  GS FX: {combined.shape}")
    return combined


def make_gs_macro_features(indices_df, fx_df=None, normalize=True):
    """Convert GS index/FX data into macro regime features.

    Derived features: VIX level, index momentum, cross-asset correlations,
    FX momentum, EM vs DM spread, commodity regime.
    """
    features = pd.DataFrame(index=indices_df.index)

    # VIX level and changes
    if "vix" in indices_df.columns:
        features["gs_vix"] = indices_df["vix"]
        features["gs_vix_change_5d"] = indices_df["vix"].pct_change(5)
        features["gs_vix_above_20"] = (indices_df["vix"] > 20).astype(float)

    # Index momentum (21-day returns)
    for col in ["spx", "ndx", "rty", "mxef", "mxwo", "mxea", "bcomtr", "dxy"]:
        if col in indices_df.columns:
            features[f"gs_{col}_mom_21d"] = indices_df[col].pct_change(21)

    # EM vs DM spread (MXEF vs MXWO momentum differential)
    if "mxef" in indices_df.columns and "mxwo" in indices_df.columns:
        em_mom = indices_df["mxef"].pct_change(21)
        dm_mom = indices_df["mxwo"].pct_change(21)
        features["gs_em_dm_spread"] = em_mom - dm_mom

    # Equity-commodity divergence
    if "spx" in indices_df.columns and "bcomtr" in indices_df.columns:
        features["gs_eq_cmd_spread"] = (
            indices_df["spx"].pct_change(21) - indices_df["bcomtr"].pct_change(21)
        )

    # DXY regime
    if "dxy" in indices_df.columns:
        features["gs_dxy_mom_5d"] = indices_df["dxy"].pct_change(5)

    # FX features
    if fx_df is not None and not fx_df.empty:
        for col in fx_df.columns:
            features[f"gs_{col}_mom_5d"] = fx_df[col].pct_change(5)

    features = features.ffill().bfill().fillna(0)

    if normalize:
        rolling_mean = features.rolling(252, min_periods=60).mean()
        rolling_std = features.rolling(252, min_periods=60).std()
        features = (features - rolling_mean) / (rolling_std + 1e-8)
        features = features.clip(-3, 3).fillna(0)

    return features


def load_gs_eod_data(start, end):
    """Load daily spot, market cap, ADV from GSEOD dataset.

    Provides liquidity and market-breadth features unavailable from TREOD.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"gs_eod_{start}_{end}.csv")
    if os.path.exists(cp):
        df = pd.read_csv(cp, index_col=0, parse_dates=True)
        print(f"  GS EOD: {df.shape} (cached)")
        return df

    session = _get_session()
    if session is None:
        return pd.DataFrame()

    from gs_quant.api.gs.data import GsDataApi

    asset_ids = _resolve_assets(GS_INDEX_TICKERS, id_type="ticker")
    if not asset_ids:
        return pd.DataFrame()

    frames = {}
    for name, mid in asset_ids.items():
        try:
            rows = GsDataApi.query_data(
                query={
                    "where": {"assetId": [mid]},
                    "startDate": str(start),
                    "endDate": str(end),
                },
                dataset_id="GSEOD",
            )
            if rows:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                # Extract useful columns
                for col in ["spot", "mdv22Day", "adv"]:
                    if col in df.columns:
                        frames[f"{name}_{col}"] = df[col]
                print(f"    {name} GSEOD: {len(df)} days")
        except Exception as e:
            print(f"    {name} GSEOD: {str(e)[:60]}")

    if not frames:
        return pd.DataFrame()

    combined = pd.DataFrame(frames).sort_index().ffill().bfill()
    combined.to_csv(cp)
    print(f"  GS EOD: {combined.shape}")
    return combined


def load_gs_fxvol_data(start, end):
    """Load FX implied volatility from FXIMPLIEDVOL_STANDARD.

    Provides vol regime features for currency-sensitive universes.
    Extracts ATM 1m vol for major pairs.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"gs_fxvol_{start}_{end}.csv")
    if os.path.exists(cp):
        df = pd.read_csv(cp, index_col=0, parse_dates=True)
        print(f"  GS FX Vol: {df.shape} (cached)")
        return df

    session = _get_session()
    if session is None:
        return pd.DataFrame()

    from gs_quant.api.gs.data import GsDataApi

    fx_ids = _resolve_assets(GS_FX_PAIRS, id_type="bbg")
    if not fx_ids:
        return pd.DataFrame()

    frames = {}
    for name, mid in fx_ids.items():
        try:
            rows = GsDataApi.query_data(
                query={
                    "where": {
                        "assetId": [mid],
                        "tenor": ["1m"],
                        "deltaStrike": ["ATMS"],
                    },
                    "startDate": str(start),
                    "endDate": str(end),
                },
                dataset_id="FXIMPLIEDVOL_STANDARD",
            )
            if rows:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"])
                series = df.set_index("date")["impliedVolatility"].sort_index()
                series.name = f"{name}_iv_1m"
                frames[name] = series
                print(f"    {name} FX IV: {len(series)} days")
        except Exception as e:
            print(f"    {name} FX IV: {str(e)[:60]}")

    if not frames:
        return pd.DataFrame()

    combined = pd.DataFrame(frames).sort_index().ffill().bfill()
    combined.to_csv(cp)
    print(f"  GS FX Vol: {combined.shape}")
    return combined


def load_gs_data(start, end):
    """Load all available GS Quant data and return macro feature DataFrame.

    Pulls from: TREOD (13 indices), FXSPOT (2 pairs), GSEOD (market cap/ADV),
    FXIMPLIEDVOL (FX vol surface).
    """
    print("Loading GS Quant data...")
    indices = load_gs_index_data(start, end)
    fx = load_gs_fx_data(start, end)

    if indices.empty and fx.empty:
        print("  No GS Quant data available.")
        return pd.DataFrame()

    features = make_gs_macro_features(indices, fx) if not indices.empty else pd.DataFrame()

    # Add GSEOD market microstructure features
    eod = load_gs_eod_data(start, end)
    if not eod.empty:
        if not features.empty:
            features = pd.concat([features, eod], axis=1, join="outer").ffill().bfill()
        else:
            features = eod

    # Add FX implied vol
    fxvol = load_gs_fxvol_data(start, end)
    if not fxvol.empty:
        if not features.empty:
            features = pd.concat([features, fxvol], axis=1, join="outer").ffill().bfill()
        else:
            features = fxvol

    print(f"  GS Quant total features: {features.shape}")
    return features

# ====================================================================
# Module: llm_features.py
# ====================================================================
"""LLM feature extraction: FinBERT embeddings + Qwen3-8B structured sentiment.

Designed to run on Google Colab free T4 (15GB VRAM):
- FinBERT: ~2GB VRAM, batch inference
- Qwen3-8B (4-bit): ~5GB VRAM, structured JSON output
"""

import json
import os

import numpy as np


def get_gemini_embeddings(headlines, model="text-embedding-004", batch_size=100):
    """Extract embeddings via Google Gemini API.

    Uses GOOGLE_API_KEY from .env.  Returns shape (len(headlines), 768).
    Falls back to zeros if the API is unavailable.

    Args:
        headlines: list of headline strings
        model: Gemini embedding model name
        batch_size: headlines per API call
    Returns:
        np.ndarray of shape (len(headlines), 768)
    """
    import os
    import time

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not set. Returning zero embeddings.")
        return np.zeros((len(headlines), 768), dtype=np.float32)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
    except ImportError:
        print("Install google-generativeai: pip install google-generativeai")
        return np.zeros((len(headlines), 768), dtype=np.float32)

    embeddings = []
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        for attempt in range(3):
            try:
                result = genai.embed_content(
                    model=f"models/{model}",
                    content=batch,
                    task_type="SEMANTIC_SIMILARITY",
                )
                emb = np.array(result["embedding"], dtype=np.float32)
                # Gemini may return different dims; truncate/pad to 768
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                if emb.shape[1] > 768:
                    emb = emb[:, :768]
                elif emb.shape[1] < 768:
                    pad = np.zeros((emb.shape[0], 768 - emb.shape[1]), dtype=np.float32)
                    emb = np.hstack([emb, pad])
                embeddings.append(emb)
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait = 2 ** (attempt + 1)
                    print(f"  Gemini rate limit, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Gemini embedding error: {e}")
                    embeddings.append(np.zeros((len(batch), 768), dtype=np.float32))
                    break

    if embeddings:
        return np.vstack(embeddings)
    return np.zeros((len(headlines), 768), dtype=np.float32)


def apply_soft_zca_whitening(embeddings, alpha=0.05):
    """Apply Soft-ZCA whitening to fix BERT anisotropy (cosine sim collapse).

    Reference: "Isotropy Matters: Soft-ZCA Whitening of Embeddings"
    (arXiv 2411.17538). BERT mean-pooled embeddings cluster into a narrow
    cone (cosine sim ~1.0). Whitening restores isotropy and recovers
    discriminative signal.

    Args:
        embeddings: (N, D) array of raw BERT embeddings
        alpha: regularization strength (0=full ZCA, 1=identity); 0.05 is
               the documented sweet spot from the paper
    Returns:
        (N, D) whitened embeddings with restored variance
    """
    if embeddings.shape[0] < 2:
        return embeddings  # need ≥2 samples to compute covariance

    # Center
    mu = embeddings.mean(axis=0, keepdims=True)
    X = embeddings - mu

    # Covariance + regularized eigendecomposition
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-8)

    # Soft ZCA: blend whitening with identity by alpha
    # W = U * diag(eigvals^{-(1-alpha)/2}) * U^T
    inv_sqrt = eigvals ** (-(1 - alpha) / 2)
    W = eigvecs @ np.diag(inv_sqrt) @ eigvecs.T

    return (X @ W).astype(np.float32)


def get_finbert_embeddings(headlines, batch_size=None, device="cuda",
                            apply_whitening=True, pooling="mean"):
    """Extract 768-dim FinBERT embeddings from financial headlines.

    By default applies Soft-ZCA whitening to fix the documented BERT
    anisotropy problem (without whitening, all embeddings have cosine
    similarity ~1.0 — see arXiv 2411.17538).

    Batch size auto-scales: 256 on A100 (80GB), 64 on T4 (16GB).

    Args:
        headlines: list of headline strings
        batch_size: inference batch size (None = auto by VRAM)
        device: "cuda" or "cpu"
        apply_whitening: if True, apply Soft-ZCA whitening (recommended)
        pooling: "mean" (default), "cls", or "max"
    Returns:
        np.ndarray of shape (len(headlines), 768)
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    if batch_size is None:
        if device == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
            batch_size = 256 if vram_gb >= 40 else 64
        else:
            batch_size = 32

    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        if pooling == "cls":
            emb = outputs.last_hidden_state[:, 0, :]
        elif pooling == "max":
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            hidden = outputs.last_hidden_state * mask + (1 - mask) * (-1e9)
            emb = hidden.max(dim=1).values
        else:  # mean
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

        embeddings.append(emb.cpu().numpy())

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    raw = np.vstack(embeddings) if embeddings else np.empty((0, 768))

    if apply_whitening and raw.shape[0] >= 2:
        return apply_soft_zca_whitening(raw, alpha=0.05)
    return raw


def build_temporal_text_tensor(
    sample_dates,
    universe_tickers,
    headlines_df,
    headline_to_emb,
    lookback_days=60,
    min_headlines=3,
    train_end=None,
):
    """Build a time-varying per-asset text tensor for training and backtest.

    Fixes the critical bug where `np.tile(prior_embs, (n_samp, 1, 1))` produced
    identical text features at every timestep — making per-timestep cosine
    similarity collapse to 1.0 regardless of embedding model.

    For each sample date t and each asset a, average headline embeddings from
    the window [t - lookback_days, t] for that asset. If fewer than
    min_headlines, fall back in priority:
      1. Asset's embedding from the previous successful date (temporal smoothing)
      2. Market-wide average over the same window
      3. All-time market average

    Args:
        sample_dates: pd.DatetimeIndex of length n_samp (from build_dataset
                      with return_dates=True)
        universe_tickers: dict like UNIVERSES['DM']; ordered ticker list
        headlines_df: DataFrame with columns [date, ticker, headline]
        headline_to_emb: dict mapping headline string -> np.ndarray embedding
        lookback_days: rolling window size for text aggregation
        min_headlines: minimum headlines in window to avoid fallback
        train_end: if set, use prior-only text for dates after train_end
                   (prevents look-ahead in OOS backtest)
    Returns:
        train_tensor: (n_samp, n_assets, emb_dim) np.float32
        pit_dict: {pd.Timestamp: (n_assets, emb_dim)} for date-keyed lookup
    """
    import pandas as pd

    if not headline_to_emb:
        return None, {}

    emb_dim = next(iter(headline_to_emb.values())).shape[-1]
    ticker_list = list(universe_tickers.values())
    n_assets = len(ticker_list)
    n_samp = len(sample_dates)

    hdl_df = headlines_df.copy()
    if "date" in hdl_df.columns:
        hdl_df["date"] = pd.to_datetime(hdl_df["date"], errors="coerce")
    else:
        hdl_df["date"] = pd.NaT

    # Global market fallback (all-time average)
    all_embs = list(headline_to_emb.values())
    market_fallback = (
        np.mean(all_embs, axis=0).astype(np.float32)
        if all_embs
        else np.zeros(emb_dim, dtype=np.float32)
    )

    cutoff = pd.Timestamp(train_end) if train_end is not None else None

    def _window_emb(ticker, date_end, market_window_emb):
        """Mean embedding for a ticker's headlines in [date_end - lookback, date_end]."""
        window_start = date_end - pd.Timedelta(days=lookback_days)
        mask = (
            (hdl_df["ticker"] == ticker)
            & (hdl_df["date"].notna())
            & (hdl_df["date"] >= window_start)
            & (hdl_df["date"] <= date_end)
        )
        rows = hdl_df[mask]["headline"].tolist()
        embs = [headline_to_emb[h] for h in rows if h in headline_to_emb]
        if len(embs) >= min_headlines:
            return np.mean(embs, axis=0).astype(np.float32)
        return market_window_emb  # fallback

    def _market_window_emb(date_end):
        window_start = date_end - pd.Timedelta(days=lookback_days)
        mask = (
            hdl_df["date"].notna()
            & (hdl_df["date"] >= window_start)
            & (hdl_df["date"] <= date_end)
        )
        rows = hdl_df[mask]["headline"].tolist()
        embs = [headline_to_emb[h] for h in rows if h in headline_to_emb]
        if len(embs) >= min_headlines:
            return np.mean(embs, axis=0).astype(np.float32)
        return market_fallback

    # Training tensor: for each sample date, use prior-only text (≤ train_end
    # if specified, else ≤ sample_date itself — but sample dates are already
    # constrained to <= train_end by build_dataset when train_end is passed).
    train_tensor = np.zeros((n_samp, n_assets, emb_dim), dtype=np.float32)
    prev_per_asset = [None] * n_assets  # temporal smoothing carry
    coverage = np.zeros(n_assets, dtype=int)

    for i, dt in enumerate(sample_dates):
        effective_date = dt
        if cutoff is not None and dt > cutoff:
            effective_date = cutoff  # clamp to prevent look-ahead during training
        mkt = _market_window_emb(effective_date)
        for j, ticker in enumerate(ticker_list):
            emb = _window_emb(ticker, effective_date, mkt)
            # Prefer per-asset fresh > per-asset previous > market fallback
            if np.allclose(emb, mkt) and prev_per_asset[j] is not None:
                train_tensor[i, j] = prev_per_asset[j]
            else:
                train_tensor[i, j] = emb
                if not np.allclose(emb, mkt):
                    prev_per_asset[j] = emb
                    coverage[j] += 1

    # Point-in-time dict for backtest: lookup by date, uses true prior window
    pit_dict = {}
    for dt in sample_dates:
        ts = pd.Timestamp(dt)
        mkt = _market_window_emb(ts)
        assets = np.zeros((n_assets, emb_dim), dtype=np.float32)
        for j, ticker in enumerate(ticker_list):
            assets[j] = _window_emb(ticker, ts, mkt)
        pit_dict[ts] = assets

    print(
        f"    Temporal coverage: min={coverage.min()} max={coverage.max()} "
        f"mean={coverage.mean():.0f} samples-with-own-headlines per asset "
        f"(lookback={lookback_days}d, min={min_headlines} headlines)"
    )
    return train_tensor, pit_dict


def aggregate_text_per_timestep(fb, method="norm_mean_max_concat"):
    """Aggregate per-asset text embeddings into a per-timestep feature vector.

    Critical fix for the anisotropy collapse: simple mean across n_assets
    of high-dim BERT embeddings collapses to ~zero, making every timestep
    look identical. Instead, L2-normalize each asset embedding first, then
    use [mean, max] concatenation to preserve per-asset variance signal.

    Args:
        fb: (n_dates, n_assets, D) per-asset text embeddings
        method:
            "mean" — legacy (broken): simple mean across assets, returns (N, D)
            "norm_mean" — L2-normalize first, then mean. Returns (N, D)
            "norm_mean_max_concat" — L2-norm, then [mean, max] concat. Returns (N, 2D)
    Returns:
        (n_dates, D_out) timestep-level text features
    """
    if method == "mean":
        return fb.mean(axis=1).astype(np.float32)

    norms = np.linalg.norm(fb, axis=-1, keepdims=True)
    fb_norm = fb / (norms + 1e-8)

    if method == "norm_mean":
        return fb_norm.mean(axis=1).astype(np.float32)

    # Default: norm_mean_max_concat preserves variance + extreme signals
    mean_part = fb_norm.mean(axis=1)
    max_part = fb_norm.max(axis=1)
    return np.concatenate([mean_part, max_part], axis=-1).astype(np.float32)


def get_finance_sentence_embeddings(headlines, device="cuda",
                                     model_name="FinLang/finance-embeddings-investopedia"):
    """Alternative: use a finance-domain sentence-transformer.

    FinLang/finance-embeddings-investopedia is fine-tuned from BAAI/bge-base-en-v1.5
    on Investopedia finance text. Built with sentence-transformers, so it
    natively avoids the BERT anisotropy problem.

    Use this as a drop-in replacement for get_finbert_embeddings() when
    text quality matters more than backwards compatibility with the
    ProsusAI/finbert sentiment classifier.

    Args:
        headlines: list of headline strings
        device: "cuda" or "cpu"
        model_name: HuggingFace model ID (default: FinLang investopedia)
    Returns:
        np.ndarray of shape (len(headlines), 768)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers required: pip install sentence-transformers"
        )

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        headlines,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity
    )

    del model
    import torch
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings.astype(np.float32)


def _select_qwen_model():
    """Select best Qwen3 model for available VRAM.

    A100 80GB  -> Qwen3-32B  (4-bit ~20GB, best structured reasoning)
    T4/V100    -> Qwen3-8B   (4-bit ~5GB, good baseline)
    CPU        -> None        (skip)
    """
    import torch
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    vram_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
    if vram_gb >= 70:
        return "Qwen/Qwen3-32B"   # A100 80GB / H100: best reasoning + JSON
    elif vram_gb >= 12:
        return "Qwen/Qwen3-8B"    # A100 40GB / T4 / V100
    return None


def get_qwen3_sentiment(headlines_batch, device="cuda", max_new_tokens=256,
                        model_name=None):
    """Extract structured sentiment using Qwen3-32B (A100) or Qwen3-8B (T4).

    Auto-selects model based on GPU VRAM. Returns JSON with:
    sentiment (-1 to 1), risk_level, regime, key_factors, confidence.

    Args:
        headlines_batch: list of lists -- each inner list is headlines for one prompt
        device: "cuda" or "cpu"
        max_new_tokens: max generation length
        model_name: override model (None = auto-select by VRAM)
    Returns:
        list of dicts with structured sentiment data
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    if model_name is None:
        model_name = _select_qwen_model()
    if model_name is None:
        print("  No GPU available for Qwen3. Returning neutral sentiment.")
        return [{"sentiment": 0.0, "risk_level": "medium",
                 "regime": "transitioning", "key_factors": [],
                 "confidence": 0.0}] * len(headlines_batch)

    print(f"  Loading {model_name} (4-bit)...")

    # Free VRAM from any prior models (e.g. FinBERT) before loading
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"  {model_name} loaded. VRAM: {vram_used:.1f} GB")

    PROMPT_TEMPLATE = (
        "You are an expert financial analyst. Analyze these financial headlines "
        "and return ONLY valid JSON (no explanation, no markdown):\n\n"
        "Headlines:\n{headlines}\n\n"
        "Return exactly this JSON structure:\n"
        '{{"sentiment": <float -1.0 to 1.0>, '
        '"risk_level": "low" | "medium" | "high", '
        '"regime": "risk_on" | "risk_off" | "transitioning", '
        '"key_factors": [<up to 5 short strings>], '
        '"confidence": <float 0.0 to 1.0>}}'
    )

    results = []
    for headlines in headlines_batch:
        prompt = PROMPT_TEMPLATE.format(headlines="\n".join(headlines[:15]))
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.1, do_sample=True,
            )
        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Extract JSON from response (handle markdown code blocks)
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]

        try:
            parsed = json.loads(text)
            # Validate expected fields
            parsed.setdefault("sentiment", 0.0)
            parsed.setdefault("risk_level", "medium")
            parsed.setdefault("regime", "transitioning")
            parsed.setdefault("key_factors", [])
            parsed.setdefault("confidence", 0.5)
            results.append(parsed)
        except json.JSONDecodeError:
            results.append({
                "sentiment": 0.0, "risk_level": "medium",
                "regime": "transitioning", "key_factors": [],
                "confidence": 0.0,
            })

    del model
    torch.cuda.empty_cache()

    return results


def sentiment_to_features(sentiment_data, feature_dim=16):
    """Convert structured sentiment output to a fixed-size feature vector.

    Args:
        sentiment_data: dict from get_qwen3_sentiment
        feature_dim: output dimension
    Returns:
        np.ndarray of shape (feature_dim,)
    """
    feat = np.zeros(feature_dim, dtype=np.float32)

    # Sentiment score
    feat[0] = sentiment_data.get("sentiment", 0.0)

    # Risk level encoding
    risk_map = {"low": -1.0, "medium": 0.0, "high": 1.0}
    feat[1] = risk_map.get(sentiment_data.get("risk_level", "medium"), 0.0)

    # Regime encoding (one-hot)
    regime = sentiment_data.get("regime", "transitioning")
    feat[2] = 1.0 if regime == "risk_on" else 0.0
    feat[3] = 1.0 if regime == "risk_off" else 0.0
    feat[4] = 1.0 if regime == "transitioning" else 0.0

    # Number of key factors as complexity signal
    n_factors = len(sentiment_data.get("key_factors", []))
    feat[5] = min(n_factors / 5.0, 1.0)

    # Confidence (from Qwen3-32B)
    feat[6] = sentiment_data.get("confidence", 0.5)

    return feat


def build_text_features(
    headlines_by_date_ticker,
    rebalance_dates,
    tickers,
    use_finbert=True,
    use_qwen=True,
    finbert_dim=768,
    sentiment_dim=16,
    device="cuda",
):
    """Build per-asset, per-date text feature matrix.

    Args:
        headlines_by_date_ticker: dict mapping (date, ticker) -> list of headlines
        rebalance_dates: list of rebalance dates
        tickers: list of tickers
        use_finbert: whether to compute FinBERT embeddings
        use_qwen: whether to compute Qwen3 sentiment
        finbert_dim: FinBERT embedding dimension
        sentiment_dim: structured sentiment feature dimension
        device: compute device
    Returns:
        dict with keys:
            "finbert": np.ndarray (n_dates, n_assets, finbert_dim) or None
            "sentiment": np.ndarray (n_dates, n_assets, sentiment_dim) or None
    """
    n_dates = len(rebalance_dates)
    n_assets = len(tickers)

    finbert_features = np.zeros((n_dates, n_assets, finbert_dim), dtype=np.float32) if use_finbert else None
    sentiment_features = np.zeros((n_dates, n_assets, sentiment_dim), dtype=np.float32) if use_qwen else None

    for i, rd in enumerate(rebalance_dates):
        for j, ticker in enumerate(tickers):
            key = (pd.Timestamp(rd) if not isinstance(rd, pd.Timestamp) else rd, ticker)
            headlines = headlines_by_date_ticker.get(key, [])
            if not headlines:
                continue

            if use_finbert:
                embs = get_finbert_embeddings(headlines, device=device)
                finbert_features[i, j] = embs.mean(axis=0)  # Average over headlines

            if use_qwen:
                sentiments = get_qwen3_sentiment([headlines], device=device)
                if sentiments:
                    sentiment_features[i, j] = sentiment_to_features(sentiments[0], sentiment_dim)

    return {"finbert": finbert_features, "sentiment": sentiment_features}


# Avoid circular import — only import pandas when needed
import pandas as pd

# ====================================================================
# Module: price_loader.py
# ====================================================================
import json
import os
import time

import numpy as np
import pandas as pd
import requests


# Cache directory for downloaded data
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")

# Default asset universes — balanced at ~10 assets each
# GLD is in Commodities only (no overlap with DM)
UNIVERSES = {
    "DM": {
        "US": "SPY", "Nasdaq": "QQQ", "SmallCap": "IWM",
        "EAFE": "EFA", "Europe": "VGK",
        "LongBond": "TLT", "MidBond": "IEF", "CorpBond": "LQD",
        "REIT": "VNQ", "Dollar": "UUP",
    },
    "DM_small": {
        "US": "SPY", "EAFE": "EFA", "Bonds": "TLT", "Dollar": "UUP",
    },
    "EM": {
        "EM": "EEM", "Brazil": "EWZ", "China": "FXI",
        "Korea": "EWY", "Taiwan": "EWT", "India": "INDA",
        "Mexico": "EWW", "SouthAfrica": "EZA",
        "Thailand": "THD", "Turkey": "TUR",
    },
    "EM_small": {
        "EM": "EEM", "Brazil": "EWZ", "China": "FXI", "Korea": "EWY",
    },
    "Commodities": {
        "Oil": "USO", "NatGas": "UNG", "Gold": "GLD",
        "Silver": "SLV", "Agriculture": "DBA", "Commodities": "DBC",
        "Copper": "CPER", "Wheat": "WEAT", "Corn": "CORN", "Soybeans": "SOYB",
    },
    # US Sector ETFs — SPDR Select Sector SPDRs (10 sectors, deep history)
    "Sectors": {
        "Tech": "XLK", "Financials": "XLF", "HealthCare": "XLV",
        "Energy": "XLE", "ConsumerDisc": "XLY", "ConsumerStap": "XLP",
        "Industrials": "XLI", "Utilities": "XLU", "Materials": "XLB",
        "RealEstate": "XLRE",
    },
    # Global Multi-Asset — balanced 10-asset global portfolio
    "Global": {
        "USLarge": "SPY", "USGrowth": "QQQ", "EAFE": "EFA", "EM": "EEM",
        "USBonds": "AGG", "TIPS": "TIP", "HighYield": "HYG",
        "Gold": "GLD", "REIT": "VNQ", "Commodities": "DBC",
    },
    # Factor/Style ETFs — for factor-exposure analysis
    "Factors": {
        "Momentum": "MTUM", "Value": "VLUE", "Quality": "QUAL",
        "MinVol": "USMV", "Size": "SIZE", "Dividend": "VIG",
        "Growth": "IVW", "Blend": "IVV", "SmallValue": "SLYV", "SmallGrowth": "SLYG",
    },
    # Bonds — diversified fixed income across duration, credit, geography
    "Bonds": {
        "LongTreasury": "TLT", "IntermTreasury": "IEF", "ShortTreasury": "SHY",
        "Corporate": "LQD", "HighYield": "HYG", "EMBond": "EMB",
        "Muni": "MUB", "IntlBond": "BNDX",
        "FloatingRate": "FLOT", "TIPS": "TIP",
    },
    # Crypto — limited history (post-2020). Used only for recent-period analysis.
    "Crypto": {
        "BitcoinETF": "GBTC", "EthereumETF": "ETHE",
        # Blockchain-related equities as proxies for broader crypto exposure
        "Coinbase": "COIN", "MicroStrategy": "MSTR",
        "MarathonDigital": "MARA", "RiotPlatforms": "RIOT",
        "CleanSpark": "CLSK", "Hut8Mining": "HUT",
        "BitcoinFutures": "BITO", "EthFutures": "EETH",
    },
}

# Shared session with browser-like headers
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
})


def _cache_path(ticker, start, end):
    """Return cache file path for a ticker."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.csv")


def _download_yahoo_raw(ticker, start, end):
    """Download daily close prices using the Yahoo Finance v8 API directly.

    This bypasses yfinance library rate-limit issues by using a custom session.
    """
    t1 = int(pd.Timestamp(start).timestamp())
    t2 = int(pd.Timestamp(end).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={t1}&period2={t2}&interval=1d"
    )
    resp = _SESSION.get(url, timeout=15)
    if resp.status_code != 200:
        return None

    data = resp.json()
    result = data.get("chart", {}).get("result", [])
    if not result:
        return None

    timestamps = result[0].get("timestamp", [])
    closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])

    if not timestamps or not closes:
        return None

    dates = pd.to_datetime(timestamps, unit="s").normalize()
    series = pd.Series(closes, index=dates, name=ticker, dtype=np.float64)
    series = series.dropna()
    series.index.name = "Date"
    return series


def _download_yahoo_full(ticker, start, end):
    """Download daily close + volume from Yahoo v8 API."""
    t1 = int(pd.Timestamp(start).timestamp())
    t2 = int(pd.Timestamp(end).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={t1}&period2={t2}&interval=1d"
    )
    resp = _SESSION.get(url, timeout=15)
    if resp.status_code != 200:
        return None

    data = resp.json()
    result = data.get("chart", {}).get("result", [])
    if not result:
        return None

    timestamps = result[0].get("timestamp", [])
    quote = result[0].get("indicators", {}).get("quote", [{}])[0]
    closes = quote.get("close", [])
    volumes = quote.get("volume", [])

    if not timestamps or not closes:
        return None

    dates = pd.to_datetime(timestamps, unit="s").normalize()
    df = pd.DataFrame({"close": closes, "volume": volumes}, index=dates)
    df.index.name = "Date"
    return df.dropna(subset=["close"])


def load_etf_volume_data(tickers, start, end, delay=1.0):
    """Load daily traded volume for a list of tickers.

    Uses local CSV cache and falls back to Yahoo v8 API.
    """
    if isinstance(tickers, dict):
        tickers = list(tickers.values())

    frames = {}
    need_download = []

    for ticker in tickers:
        cp = os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}_vol.csv")
        os.makedirs(CACHE_DIR, exist_ok=True)
        if os.path.exists(cp):
            try:
                s = pd.read_csv(cp, index_col=0, parse_dates=True).squeeze()
                if len(s) > 100:
                    frames[ticker] = s
                    continue
            except Exception:
                pass
        need_download.append(ticker)

    if need_download:
        print(f"  Downloading volume for {len(need_download)} tickers...")
        for i, ticker in enumerate(need_download):
            for attempt in range(3):
                full = _download_yahoo_full(ticker, start, end)
                if full is not None and "volume" in full.columns and len(full) > 100:
                    vol_s = full["volume"].rename(ticker)
                    frames[ticker] = vol_s
                    cp = os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}_vol.csv")
                    vol_s.to_csv(cp)
                    break
                time.sleep(delay * (attempt + 1))
            if i < len(need_download) - 1:
                time.sleep(delay * 0.5)

    if not frames:
        return pd.DataFrame()
    data = pd.DataFrame(frames).dropna(how="all")
    print(f"  Volume: {data.shape[0]}x{data.shape[1]} assets")
    return data


def load_options_snapshot(tickers, delay=0.5):
    """Get current options snapshot (IV proxy, put/call ratio) for tickers.

    Note: point-in-time snapshot, not historical. Use for latest period analysis.
    """
    import yfinance as yf

    if isinstance(tickers, dict):
        tickers = list(tickers.values())

    results = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            exp_dates = t.options
            if not exp_dates:
                continue
            chain = t.option_chain(exp_dates[0])
            calls, puts = chain.calls, chain.puts

            last_price = None
            try:
                last_price = t.fast_info.get("lastPrice")
            except Exception:
                pass
            if last_price is None:
                continue

            call_iv = np.nan
            if not calls.empty and "impliedVolatility" in calls.columns:
                atm_idx = (calls["strike"] - last_price).abs().idxmin()
                call_iv = calls.loc[atm_idx, "impliedVolatility"]

            put_iv = np.nan
            if not puts.empty and "impliedVolatility" in puts.columns:
                atm_idx = (puts["strike"] - last_price).abs().idxmin()
                put_iv = puts.loc[atm_idx, "impliedVolatility"]

            total_call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
            total_put_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
            pc_ratio = total_put_oi / max(total_call_oi, 1)

            results[ticker] = {
                "atm_iv": np.nanmean([call_iv, put_iv]),
                "put_call_oi_ratio": pc_ratio,
            }
        except Exception:
            continue
        time.sleep(delay)

    return pd.DataFrame(results).T if results else pd.DataFrame()


def load_etf_data(tickers, start, end, delay=1.0):
    """Load adjusted close prices for a list of tickers.

    Uses local CSV cache and falls back to Yahoo v8 API.
    """
    if isinstance(tickers, dict):
        tickers = list(tickers.values())

    frames = {}
    need_download = []

    # Check cache first
    for ticker in tickers:
        cp = _cache_path(ticker, start, end)
        if os.path.exists(cp):
            try:
                s = pd.read_csv(cp, index_col=0, parse_dates=True).squeeze()
                if len(s) > 100:
                    frames[ticker] = s
                    print(f"  [cache] {ticker}: {len(s)} days")
                    continue
            except Exception:
                pass
        need_download.append(ticker)

    # Download missing tickers via raw API
    if need_download:
        print(f"  Downloading {len(need_download)} tickers...")
        for i, ticker in enumerate(need_download):
            for attempt in range(3):
                series = _download_yahoo_raw(ticker, start, end)
                if series is not None and len(series) > 100:
                    frames[ticker] = series
                    series.to_csv(_cache_path(ticker, start, end))
                    print(f"  [{i+1}/{len(need_download)}] {ticker}: {len(series)} days")
                    break
                time.sleep(delay * (attempt + 1))
            else:
                print(f"  [{i+1}/{len(need_download)}] {ticker}: FAILED")
            if i < len(need_download) - 1:
                time.sleep(delay)

    if not frames:
        print("  ERROR: No data available")
        return pd.DataFrame()

    data = pd.DataFrame(frames)
    data = data.dropna(axis=1, how="all")
    coverage = 1 - data.isna().mean().mean()
    print(f"  Total: {data.shape[0]}x{data.shape[1]} assets, coverage: {coverage:.1%}")
    return data.dropna(how="all")


def load_fama_french(start, end):
    """Load Fama-French daily factors."""
    cp = os.path.join(CACHE_DIR, f"FF3_{start}_{end}.csv")
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(cp):
        ff = pd.read_csv(cp, index_col=0, parse_dates=True)
        print(f"  FF factors: {ff.shape[0]} days (cached)")
        return ff

    import pandas_datareader.data as pdr
    ff = pdr.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start, end)[0] / 100
    ff.to_csv(cp)
    print(f"  FF factors: {ff.shape[0]} days")
    return ff


def load_universe(name, start, end):
    """Load prices for a named universe."""
    if name not in UNIVERSES:
        raise ValueError(f"Unknown universe: {name}. Available: {list(UNIVERSES.keys())}")
    tickers = UNIVERSES[name]
    return load_etf_data(tickers, start, end)

# ====================================================================
# Module: spgci_loader.py
# ====================================================================
"""S&P Global Commodity Insights (Platts) data loader.

Pulls historical commodity price assessments and forward curves via the
spgci Python SDK.  Requires SPGCI_USER and SPGCI_PASS in .env.

Access tier determines available datasets — this module fails gracefully
and returns empty DataFrames when data is unavailable.
"""

import os

import numpy as np
import pandas as pd


CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")

# Platts symbols for key commodity benchmarks
# These may need adjustment based on your subscription tier.
DEFAULT_SYMBOLS = {
    "crude_brent": "PCAAS00",   # Dated Brent (Platts)
    "crude_wti": "PCAAT00",     # WTI Midland (Platts)
    "gold": "GOLDS0Z00",       # Gold spot
    "silver": "SILVS0Z00",     # Silver spot
    "copper": "COPPS0Z00",     # Copper spot
}

DEFAULT_COMMODITY_MDCS = [
    "ET",   # Energy — crude oil
    "ME",   # Metals
]


def _authenticate():
    """Authenticate with SPGCI using credentials from .env."""
    try:
        import spgci as ci
    except ImportError:
        print("Install spgci: pip install spgci")
        return False

    user = os.environ.get("SPGCI_USER")
    passwd = os.environ.get("SPGCI_PASS")
    if not user or not passwd:
        print("Warning: SPGCI_USER/SPGCI_PASS not set.")
        return False

    try:
        ci.set_credentials(username=user, password=passwd)
        return True
    except Exception as e:
        print(f"Warning: SPGCI auth failed: {e}")
        return False


def load_spgci_assessments(start, end, symbols=None):
    """Load historical Platts commodity price assessments.

    Args:
        start: start date string
        end: end date string
        symbols: list of Platts symbol IDs, or None for defaults
    Returns:
        DataFrame with date index and one column per symbol
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"spgci_assess_{start}_{end}.csv")
    if os.path.exists(cp):
        df = pd.read_csv(cp, index_col=0, parse_dates=True)
        print(f"  SPGCI assessments: {df.shape} (cached)")
        return df

    if not _authenticate():
        return pd.DataFrame()

    import spgci as ci

    if symbols is None:
        symbols = list(DEFAULT_SYMBOLS.values())

    try:
        mdd = ci.MarketData()
        frames = {}
        for sym in symbols:
            try:
                data = mdd.get_assessments_by_symbol_historical(symbol=sym)
                if data is not None and not data.empty:
                    # Extract date and value columns
                    if "assessDate" in data.columns and "value" in data.columns:
                        ts = data.set_index("assessDate")["value"]
                        ts.index = pd.to_datetime(ts.index)
                        ts = ts[(ts.index >= start) & (ts.index <= end)]
                        if not ts.empty:
                            frames[sym] = ts
                            print(f"    {sym}: {len(ts)} observations")
            except Exception as e:
                print(f"    {sym}: not accessible ({e})")

        if frames:
            df = pd.DataFrame(frames).sort_index().ffill().bfill()
            df.to_csv(cp)
            print(f"  SPGCI assessments: {df.shape}")
            return df

    except Exception as e:
        print(f"  SPGCI assessment pull failed: {e}")

    return pd.DataFrame()


def load_spgci_forward_curves(start, end, commodities=None):
    """Load forward curve data for contango/backwardation signals.

    Args:
        start: start date string
        end: end date string
        commodities: list of commodity names, or None for defaults
    Returns:
        DataFrame with derived curve shape features
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"spgci_fwd_{start}_{end}.csv")
    if os.path.exists(cp):
        df = pd.read_csv(cp, index_col=0, parse_dates=True)
        print(f"  SPGCI forward curves: {df.shape} (cached)")
        return df

    if not _authenticate():
        return pd.DataFrame()

    if commodities is None:
        commodities = ["Crude oil"]

    import spgci as ci

    try:
        mdd = ci.MarketData()
        frames = {}
        for commodity in commodities:
            try:
                syms = mdd.get_symbols(commodity=commodity)
                if syms is not None and not syms.empty:
                    # Take first few symbols as curve points
                    for _, row in syms.head(3).iterrows():
                        sym_id = row.get("symbol", "")
                        if sym_id:
                            data = mdd.get_assessments_by_symbol_historical(
                                symbol=sym_id
                            )
                            if data is not None and not data.empty:
                                if "assessDate" in data.columns and "value" in data.columns:
                                    ts = data.set_index("assessDate")["value"]
                                    ts.index = pd.to_datetime(ts.index)
                                    ts = ts[(ts.index >= start) & (ts.index <= end)]
                                    if not ts.empty:
                                        frames[f"{commodity}_{sym_id}"] = ts
            except Exception as e:
                print(f"    Forward curve {commodity}: {e}")

        if frames:
            df = pd.DataFrame(frames).sort_index().ffill().bfill()
            df.to_csv(cp)
            print(f"  SPGCI forward curves: {df.shape}")
            return df

    except Exception as e:
        print(f"  SPGCI forward curve pull failed: {e}")

    return pd.DataFrame()


def make_spgci_features(assess_df, fwd_df=None, normalize=True):
    """Convert raw SPGCI data into feature vectors.

    Derives momentum, volatility, and curve-shape features from
    Platts assessment and forward curve data.
    """
    frames = []

    if assess_df is not None and not assess_df.empty:
        features = assess_df.copy()
        # Momentum features (21-day % change)
        for col in features.columns:
            features[f"{col}_mom_21d"] = features[col].pct_change(21)
        # Volatility features (21-day rolling std of returns)
        for col in assess_df.columns:
            ret = assess_df[col].pct_change()
            features[f"{col}_vol_21d"] = ret.rolling(21, min_periods=10).std()
        frames.append(features)

    if fwd_df is not None and not fwd_df.empty:
        frames.append(fwd_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1).ffill().bfill().fillna(0)

    if normalize:
        rolling_mean = combined.rolling(252, min_periods=60).mean()
        rolling_std = combined.rolling(252, min_periods=60).std()
        combined = (combined - rolling_mean) / (rolling_std + 1e-8)
        combined = combined.clip(-3, 3).fillna(0)

    return combined


def load_spgci_data(start, end):
    """Load all available SPGCI data and combine into feature DataFrame."""
    print("Loading SPGCI data...")
    assess = load_spgci_assessments(start, end)
    fwd = load_spgci_forward_curves(start, end)

    if assess.empty and fwd.empty:
        print("  No SPGCI data available (may need higher access tier).")
        return pd.DataFrame()

    features = make_spgci_features(assess, fwd)
    print(f"  SPGCI combined features: {features.shape}")
    return features

# ====================================================================
# Module: text_loader.py
# ====================================================================
"""Financial news headline loading for LLM feature extraction.

Sources:
- Yahoo Finance news headlines (via yfinance)
- HuggingFace datasets: financial_phrasebank, FiQA
- SEC EDGAR filings (10-K/10-Q summaries)
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def load_yfinance_news(tickers, start, end, max_headlines=100):
    """Load recent news headlines for each ticker via yfinance.

    Args:
        tickers: list of ticker symbols
        start: start date string
        end: end date string
        max_headlines: max headlines per ticker
    Returns:
        DataFrame with columns [ticker, date, headline]
    """
    import yfinance as yf

    records = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            news = t.news
            if news is None:
                continue
            for item in news[:max_headlines]:
                # Handle both old and new yfinance API formats
                content = item.get("content", item)  # new format nests under 'content'
                title = content.get("title", "") if isinstance(content, dict) else item.get("title", "")
                # Parse publication date
                pub_str = content.get("pubDate", "") if isinstance(content, dict) else ""
                if pub_str:
                    try:
                        pub_date = datetime.fromisoformat(pub_str.replace("Z", "+00:00")).strftime("%Y-%m-%d")
                    except Exception:
                        pub_date = datetime.now().strftime("%Y-%m-%d")
                elif "providerPublishTime" in item:
                    pub_date = datetime.fromtimestamp(item["providerPublishTime"]).strftime("%Y-%m-%d")
                else:
                    pub_date = datetime.now().strftime("%Y-%m-%d")
                provider = content.get("provider", {}) if isinstance(content, dict) else {}
                source = provider.get("displayName", "") if isinstance(provider, dict) else item.get("publisher", "")
                if title:
                    records.append({
                        "ticker": ticker,
                        "date": pub_date,
                        "headline": title,
                        "source": source,
                    })
        except Exception:
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
    return df


def load_rss_headlines(tickers=None, max_per_feed=50):
    """Load financial headlines from public RSS feeds.

    Sources: Reuters, CNBC, MarketWatch, Yahoo Finance RSS.
    Returns DataFrame with [ticker, date, headline, source].
    """
    try:
        import feedparser
    except ImportError:
        print("Install feedparser: pip install feedparser")
        return pd.DataFrame(columns=["ticker", "date", "headline", "source"])

    feeds = {
        "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
        "CNBC Top News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    }

    records = []
    for source_name, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = entry.get("title", "")
                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub:
                    pub_date = datetime(*pub[:6]).strftime("%Y-%m-%d")
                else:
                    pub_date = datetime.now().strftime("%Y-%m-%d")

                if not title:
                    continue

                # Map headline to tickers via keyword matching
                matched_tickers = _match_headline_to_tickers(title, tickers) if tickers else ["MARKET"]
                for ticker in matched_tickers:
                    records.append({
                        "ticker": ticker,
                        "date": pub_date,
                        "headline": title,
                        "source": source_name,
                    })
        except Exception:
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _match_headline_to_tickers(headline, tickers):
    """Match a headline to relevant tickers via keyword matching."""
    if tickers is None:
        return ["MARKET"]

    headline_upper = headline.upper()
    # Keyword mappings for common ETFs
    keywords = {
        "SPY": ["S&P", "SPY", "SP500", "S&P 500"],
        "QQQ": ["NASDAQ", "QQQ", "TECH STOCKS"],
        "EFA": ["INTERNATIONAL", "EAFE", "EUROPE", "DEVELOPED"],
        "EEM": ["EMERGING", "EM MARKET"],
        "GLD": ["GOLD", "PRECIOUS METAL"],
        "USO": ["OIL", "CRUDE", "PETROLEUM", "OPEC"],
        "TLT": ["TREASURY", "BOND", "YIELD", "RATE"],
        "UNG": ["NATURAL GAS", "NATGAS"],
        "EWZ": ["BRAZIL"],
        "FXI": ["CHINA", "CHINESE"],
        "EWY": ["KOREA", "KOREAN"],
        "INDA": ["INDIA", "INDIAN"],
        "XLF": ["BANK", "FINANCIAL"],
        "XLK": ["TECH", "TECHNOLOGY", "SOFTWARE"],
        "XLE": ["ENERGY", "OIL"],
        "DBA": ["AGRICULTURE", "CROP", "FARM"],
        "SLV": ["SILVER"],
        "CPER": ["COPPER"],
        "WEAT": ["WHEAT"],
        "CORN": ["CORN"],
    }

    matched = []
    tickers_list = list(tickers.values()) if isinstance(tickers, dict) else list(tickers)
    for ticker in tickers_list:
        if ticker in headline_upper:
            matched.append(ticker)
        elif ticker in keywords:
            for kw in keywords[ticker]:
                if kw in headline_upper:
                    matched.append(ticker)
                    break

    return matched if matched else ["MARKET"]


def load_all_headlines(tickers, start, end, max_headlines=100, use_rss=True,
                       use_phrasebank=True, use_fiqa=True, use_gdelt=True,
                       gdelt_max_per_ticker=500, gdelt_chunk_months=3):
    """Load headlines from all available sources and combine.

    Target: 5000+ unique dated headlines for robust FinBERT embeddings.

    Sources (in order of historical coverage):
      1. GDELT DOC 2.0 — 10-year historical coverage, free, no API key
      2. yfinance — recent per-ticker news
      3. RSS feeds — current headlines from Reuters, CNBC, MarketWatch
      4. Financial PhraseBank — static academic dataset (undated)
      5. FiQA — static financial QA sentiment dataset (undated)
    """
    all_dfs = []

    # GDELT historical headlines (primary source for dated coverage)
    if use_gdelt:
        print("  Loading GDELT historical headlines...")
        try:
            from src.data.gdelt_loader import load_gdelt_headlines
            gdelt_df = load_gdelt_headlines(
                tickers, start, end,
                max_per_ticker=gdelt_max_per_ticker,
                chunk_months=gdelt_chunk_months,
            )
            if not gdelt_df.empty:
                # Standardize columns
                if "tone" in gdelt_df.columns:
                    gdelt_df = gdelt_df.drop(columns=["tone"])
                print(f"    GDELT: {len(gdelt_df)} headlines")
                all_dfs.append(gdelt_df)
        except Exception as e:
            print(f"    GDELT failed: {e}")

    # Yahoo Finance per-ticker news
    print("  Loading yfinance headlines...")
    yf_df = load_yfinance_news(tickers, start, end, max_headlines=max_headlines)
    if not yf_df.empty:
        print(f"    yfinance: {len(yf_df)} headlines")
        all_dfs.append(yf_df)

    # RSS feeds
    if use_rss:
        print("  Loading RSS headlines...")
        rss_df = load_rss_headlines(tickers)
        if not rss_df.empty:
            print(f"    RSS: {len(rss_df)} headlines")
            all_dfs.append(rss_df)

    # Financial PhraseBank (static dataset — generic financial sentences)
    if use_phrasebank:
        print("  Loading Financial PhraseBank...")
        fpb = load_financial_phrasebank()
        if not fpb.empty:
            fpb_df = pd.DataFrame({
                "ticker": "MARKET",
                "date": pd.Timestamp.now(),
                "headline": fpb["sentence"],
                "source": "FinancialPhraseBank",
            })
            print(f"    PhraseBank: {len(fpb_df)} sentences")
            all_dfs.append(fpb_df)

    # FiQA sentiment dataset
    if use_fiqa:
        print("  Loading FiQA dataset...")
        fiqa = load_fiqa_sentiment()
        if not fiqa.empty:
            fiqa_df = pd.DataFrame({
                "ticker": "MARKET",
                "date": pd.Timestamp.now(),
                "headline": fiqa["sentence"],
                "source": "FiQA",
            })
            print(f"    FiQA: {len(fiqa_df)} sentences")
            all_dfs.append(fiqa_df)

    if not all_dfs:
        return pd.DataFrame(columns=["ticker", "date", "headline", "source"])

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["headline"])
    print(f"  Total unique headlines: {len(combined)}")
    return combined


def load_financial_phrasebank(split="sentences_allagree"):
    """Load the Financial PhraseBank dataset from HuggingFace.

    Args:
        split: agreement level — "sentences_allagree", "sentences_75agree", etc.
    Returns:
        DataFrame with columns [sentence, label]
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("financial_phrasebank", split)
        df = pd.DataFrame(ds["train"])
        df.columns = ["sentence", "label"]
        return df
    except Exception as e:
        print(f"Financial PhraseBank unavailable: {e}")
        return pd.DataFrame(columns=["sentence", "label"])


def load_fiqa_sentiment():
    """Load the FiQA sentiment dataset from HuggingFace.

    Returns:
        DataFrame with columns [sentence, sentiment_score]
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("pauri32/fiqa-2018")
        records = []
        for split_name in ["train", "validation", "test"]:
            if split_name in ds:
                for row in ds[split_name]:
                    records.append({
                        "sentence": row.get("sentence", ""),
                        "sentiment_score": row.get("sentiment_score", 0.0),
                    })
        return pd.DataFrame(records)
    except ImportError:
        print("Install datasets: pip install datasets")
        return pd.DataFrame(columns=["sentence", "sentiment_score"])


def aggregate_headlines_for_rebalance(
    headlines_df, rebalance_dates, tickers, lookback_days=30,
):
    """Aggregate headlines into per-asset, per-rebalance-date groups.

    For each (rebalance_date, ticker), collects all headlines from the
    lookback window. This creates the input for LLM feature extraction.

    Args:
        headlines_df: DataFrame with [ticker, date, headline]
        rebalance_dates: list of rebalance dates
        tickers: list of tickers
        lookback_days: number of days to look back for headlines
    Returns:
        dict mapping (date, ticker) -> list of headline strings
    """
    if headlines_df.empty:
        return {}

    headlines_df = headlines_df.copy()
    headlines_df["date"] = pd.to_datetime(headlines_df["date"])
    result = {}

    for rd in rebalance_dates:
        rd = pd.Timestamp(rd)
        window_start = rd - timedelta(days=lookback_days)
        window = headlines_df[
            (headlines_df["date"] >= window_start)
            & (headlines_df["date"] <= rd)
        ]
        for ticker in tickers:
            ticker_headlines = window[window["ticker"] == ticker]["headline"].tolist()
            if ticker_headlines:
                result[(rd, ticker)] = ticker_headlines

    return result


def create_text_features_placeholder(n_dates, n_assets, text_dim=768):
    """Create zero-valued text feature placeholder when no headlines available.

    Used for ablation A1 (price-only, no LLM features).
    """
    return np.zeros((n_dates, n_assets, text_dim), dtype=np.float32)

# ====================================================================
# Module: universe_config.py
# ====================================================================
"""Per-universe data modality and hyperparameter configuration.

Defines which data sources each asset-class universe receives during
LLM-DHRP training and backtesting, plus per-universe training and
backtest knobs. The goal is to let Commodities use a shorter lookback,
weekly rebalancing, stronger covariance shrinkage, and a more permissive
text gate without touching DM/EM behavior.

Defaults for DM and EM match the pre-existing hard-coded constants, so
adding these fields does not change current DM/EM results.
"""

UNIVERSE_DATA_CONFIG = {
    "DM": {
        "use_text": True,
        "use_macro": True,
        "macro_source": "fred_us",
        "lookback_window": 252,
        "rebalance_freq": 21,
        # Relaxed text suppression: gate_bias -1.0 → sigmoid≈0.27 (was -2.0 → 0.12)
        # Higher text_lr_scale so text pathway can actually learn signal
        # Lower modality_dropout so text is seen more often during training
        "text_lr_scale": 0.5,
        "modality_dropout": 0.1,
        "gate_bias_init": -1.0,
        "hrp_lam_start": 0.3,
        "hrp_lam_end": 0.05,
        "cov_shrinkage": 1e-6,
        # depth=2 best in Cell 16 ablation (Sharpe 0.268 vs depth=3 0.163)
        "tree_depth": 2,
    },
    "EM": {
        "use_text": True,
        "use_macro": True,
        "macro_source": "fred_us",
        "lookback_window": 252,
        "rebalance_freq": 21,
        "text_lr_scale": 0.5,
        "modality_dropout": 0.1,
        "gate_bias_init": -1.0,
        "hrp_lam_start": 0.3,
        "hrp_lam_end": 0.05,
        "cov_shrinkage": 0.001,  # matches backtest.py is_em branch
        # depth=2 default — same rationale as DM, ablation should re-validate
        "tree_depth": 2,
    },
    "Commodities": {
        "use_text": True,
        "use_macro": True,
        "macro_source": "fred_us",
        # Commodity-specific tuning (see plans/merry-crunching-river.md):
        # shorter regime half-life, weekly rebal to capture the
        # news-sentiment premium (Yeguang 2025, JFM), looser text gate,
        # stronger HRP anchor + covariance shrinkage for rank-deficient
        # agriculture/metals block.
        "lookback_window": 63,
        "rebalance_freq": 5,
        "text_lr_scale": 1.0,
        "modality_dropout": 0.05,
        "gate_bias_init": -0.5,
        "hrp_lam_start": 0.5,
        "hrp_lam_end": 0.1,
        "cov_shrinkage": 0.01,
        "tree_depth": 2,
    },
    # US sectors — similar to DM but with stronger within-market correlation.
    # Tree depth 2 (matches DM/EM ablation finding); deeper trees overfit on 10 assets.
    "Sectors": {
        "use_text": True,
        "use_macro": True,
        "macro_source": "fred_us",
        "lookback_window": 252,
        "rebalance_freq": 21,
        "text_lr_scale": 0.5,
        "modality_dropout": 0.1,
        "gate_bias_init": -1.0,
        "hrp_lam_start": 0.3,
        "hrp_lam_end": 0.05,
        "cov_shrinkage": 1e-6,
        "tree_depth": 2,
    },
    # Global multi-asset — diverse correlation structure, needs deeper tree
    "Global": {
        "use_text": True,
        "use_macro": True,
        "macro_source": "fred_us",
        "lookback_window": 252,
        "rebalance_freq": 21,
        "text_lr_scale": 0.5,
        "modality_dropout": 0.1,
        "gate_bias_init": -1.0,
        "hrp_lam_start": 0.3,
        "hrp_lam_end": 0.05,
        "cov_shrinkage": 1e-5,
        "tree_depth": 3,
    },
    # Factor ETFs — similar regime structure to DM
    "Factors": {
        "use_text": False,  # Factor ETFs don't have clear news attribution
        "use_macro": True,
        "macro_source": "fred_us",
        "lookback_window": 252,
        "rebalance_freq": 21,
        "text_lr_scale": 0.5,
        "modality_dropout": 0.1,
        "gate_bias_init": -1.0,
        "hrp_lam_start": 0.3,
        "hrp_lam_end": 0.05,
        "cov_shrinkage": 1e-5,
        "tree_depth": 3,
    },
    # Bonds — fixed income, low volatility, longer rebalance OK
    "Bonds": {
        "use_text": False,  # Bond news harder to attribute per-asset
        "use_macro": True,
        "macro_source": "fred_us",
        "lookback_window": 252,
        "rebalance_freq": 21,
        "text_lr_scale": 0.5,
        "modality_dropout": 0.1,
        "gate_bias_init": -1.0,
        "hrp_lam_start": 0.3,
        "hrp_lam_end": 0.05,
        "cov_shrinkage": 1e-5,
        "tree_depth": 2,
    },
    # Crypto — high volatility, short history, use short lookback
    "Crypto": {
        "use_text": True,
        "use_macro": True,
        "macro_source": "fred_us",
        "lookback_window": 63,
        "rebalance_freq": 5,
        "text_lr_scale": 1.0,  # crypto-sentiment correlation is strong
        "modality_dropout": 0.05,
        "gate_bias_init": -0.5,
        "hrp_lam_start": 0.3,
        "hrp_lam_end": 0.05,
        "cov_shrinkage": 0.02,  # Higher shrinkage for extreme volatility
        "tree_depth": 2,
    },
}


def get_universe_config(universe):
    """Look up a universe config dict, falling back to DM defaults."""
    if universe is None:
        return UNIVERSE_DATA_CONFIG["DM"]
    return UNIVERSE_DATA_CONFIG.get(universe, UNIVERSE_DATA_CONFIG["DM"])

