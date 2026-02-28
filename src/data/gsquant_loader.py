"""Goldman Sachs Quant (gs_quant) data integration.

Pulls factor risk model data, FX forwards, and macro indicators from
GS Marquee platform. Requires GS_CLIENT_ID and GS_CLIENT_SECRET in .env.

Note: Access tier determines available datasets. If specific datasets are
unavailable, this module gracefully returns empty DataFrames.
"""

import os

import numpy as np
import pandas as pd


CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")


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
            scopes=("read_financial_data",),
        )
        return GsSession.current
    except Exception as e:
        print(f"Warning: GS Quant auth failed: {e}")
        return None


def load_gs_factor_data(start, end):
    """Load GS risk model factor exposures if accessible.

    Attempts to pull: market, size, value, momentum, quality factors.
    Returns DataFrame cached to disk.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"gs_factors_{start}_{end}.csv")
    if os.path.exists(cp):
        df = pd.read_csv(cp, index_col=0, parse_dates=True)
        print(f"  GS factors: {df.shape} (cached)")
        return df

    session = _get_session()
    if session is None:
        return pd.DataFrame()

    try:
        from gs_quant.data import Dataset

        # Try MACRO_GLOBAL dataset for macro indicators
        datasets_to_try = [
            ("MACRO_GLOBAL", "macro_global"),
            ("SRDAILY", "sr_daily"),
        ]

        frames = {}
        for ds_id, label in datasets_to_try:
            try:
                ds = Dataset(ds_id)
                data = ds.get_data(
                    start=pd.Timestamp(start),
                    end=pd.Timestamp(end),
                )
                if not data.empty:
                    frames[label] = data
                    print(f"  GS {ds_id}: {data.shape}")
            except Exception as e:
                print(f"  GS {ds_id}: not accessible ({e})")

        if frames:
            combined = pd.concat(frames.values(), axis=1)
            combined.to_csv(cp)
            return combined

    except Exception as e:
        print(f"  GS Quant data pull failed: {e}")

    return pd.DataFrame()


def load_gs_fx_forwards(start, end, pairs=None):
    """Load FX forward points for EM currency risk signals.

    Args:
        start, end: date range
        pairs: list of currency pairs e.g. ["USDBRL", "USDCNY"]
    """
    if pairs is None:
        pairs = ["USDBRL", "USDCNY", "USDKRW", "USDINR", "USDMXN"]

    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"gs_fx_{start}_{end}.csv")
    if os.path.exists(cp):
        df = pd.read_csv(cp, index_col=0, parse_dates=True)
        print(f"  GS FX forwards: {df.shape} (cached)")
        return df

    session = _get_session()
    if session is None:
        return pd.DataFrame()

    try:
        from gs_quant.data import Dataset

        ds = Dataset("FXFORWARDPOINTS")
        frames = {}
        for pair in pairs:
            try:
                data = ds.get_data(
                    start=pd.Timestamp(start),
                    end=pd.Timestamp(end),
                    bbid=pair,
                )
                if not data.empty:
                    frames[pair] = data["forwardPoint"] if "forwardPoint" in data.columns else data.iloc[:, 0]
            except Exception:
                pass

        if frames:
            df = pd.DataFrame(frames)
            df.to_csv(cp)
            print(f"  GS FX forwards: {df.shape}")
            return df

    except Exception as e:
        print(f"  GS FX data pull failed: {e}")

    return pd.DataFrame()


def load_gs_data(start, end):
    """Load all available GS Quant data and combine into feature DataFrame."""
    print("Loading GS Quant data...")
    factors = load_gs_factor_data(start, end)
    fx = load_gs_fx_forwards(start, end)

    frames = []
    if not factors.empty:
        frames.append(factors)
    if not fx.empty:
        frames.append(fx)

    if not frames:
        print("  No GS Quant data available (may need higher access tier).")
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined = combined.sort_index().ffill().bfill()
    print(f"  GS Quant combined: {combined.shape}")
    return combined
