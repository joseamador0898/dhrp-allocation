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


def load_gs_data(start, end):
    """Load all available GS Quant data and return macro feature DataFrame."""
    print("Loading GS Quant data...")
    indices = load_gs_index_data(start, end)
    fx = load_gs_fx_data(start, end)

    if indices.empty and fx.empty:
        print("  No GS Quant data available.")
        return pd.DataFrame()

    if not indices.empty:
        features = make_gs_macro_features(indices, fx)
        print(f"  GS Quant macro features: {features.shape}")
        return features

    return pd.DataFrame()
