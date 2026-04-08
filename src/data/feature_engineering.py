import numpy as np
import pandas as pd

from ..models.baselines import hrp_allocation

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
                   train_end=None):
    """Build training dataset: features, covariances, forward returns, HRP weights.

    Args:
        prices: DataFrame of adjusted close prices
        window: lookback window in trading days
        step: step size between samples
        is_em: emerging markets flag
        volume: DataFrame of daily volume (same date index as prices), or None
        fdim: feature dimension
        train_end: if provided, only use data up to this date for training samples
    Returns:
        (X, S, R, H) tuple of numpy arrays
    """
    rets = prices.pct_change().dropna()
    X, S, R, H = [], [], [], []
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
        fwd_r = rets.iloc[t : t + fwd].mean().values
        if np.isnan(feat).any() or np.isnan(fwd_r).any():
            continue
        X.append(feat)
        S.append(cov.astype(np.float32))
        R.append(fwd_r.astype(np.float32))
        H.append(hrp_w.astype(np.float32))

    if X:
        return np.stack(X), np.stack(S), np.stack(R), np.stack(H)
    return np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,))
