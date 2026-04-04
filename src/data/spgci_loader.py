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
