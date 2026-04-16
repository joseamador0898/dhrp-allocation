import json
import os
import time

import numpy as np
import pandas as pd
import requests
import pandas_datareader.data as pdr


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
