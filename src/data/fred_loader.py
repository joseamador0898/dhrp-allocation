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
