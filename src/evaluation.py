"""src.evaluation (consolidated). Original layout: ['backtest.py', 'factor_analysis.py', 'statistics.py']"""

# ====================================================================
# Module: backtest.py
# ====================================================================
"""Rolling-window backtest engine supporting all portfolio methods.

Supports:
- Classical baselines: EW, MINVAR, MV, HRP, RP, MAXDIV
- DHRP (differentiable HRP)
- LLM-DHRP (LLM-enhanced differentiable HRP)
- Deep baselines: MLP, Transformer, PPO
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import warnings

from src.models import (
    equal_weight, min_variance, mean_variance, hrp_allocation,
    risk_parity, max_diversification, ledoit_wolf_cov,
)
from src.training import dhrp_weights
from src.data import (
    get_universe_config, compute_returns, make_features,
    aggregate_text_per_timestep, get_macro_vector,
)

# Default backtest parameters
TRAIN_DAYS = 252
TEST_DAYS = 21
STEP_DAYS = 21
MIN_COVERAGE = 0.95
PURGE_DAYS = 5  # gap between train and test to prevent look-ahead
METHODS = [
    "EW", "MINVAR", "MV", "HRP", "RP", "MAXDIV",
    "DHRP", "LLM_DHRP", "MLP", "Transformer", "PPO", "DFL",
]


def rolling_backtest(
    prices,
    is_em=False,
    dhrp_model=None,
    llm_dhrp_model=None,
    mlp_model=None,
    transformer_model=None,
    ppo_model=None,
    dfl_model=None,
    text_features=None,
    macro_features=None,
    methods=None,
    train_days=TRAIN_DAYS,
    test_days=TEST_DAYS,
    step_days=STEP_DAYS,
    min_coverage=MIN_COVERAGE,
    transaction_cost_bps=0,
    volume=None,
    purge_days=PURGE_DAYS,
    return_weights=False,
    oos_start=None,
    universe=None,
    weight_ema=0.0,
    strict_methods=None,
    return_diagnostics=False,
):
    """Run rolling-window backtest over all methods.

    Args:
        prices: DataFrame of adjusted close prices
        is_em: whether this is an emerging markets universe
        dhrp_model: trained DHRPLayer, or None
        llm_dhrp_model: trained LLMDHRPLayer, or None
        mlp_model: trained MLPWithCovPolicy, or None
        transformer_model: trained TransformerPortfolioPolicy, or None
        ppo_model: trained PPOPortfolioAgent, or None
        text_features: dict with 'finbert' and/or 'sentiment' arrays, or None
        macro_features: DataFrame of macro features, or None
        methods: list of method names to backtest
        train_days: rolling training window size
        test_days: forward test window size
        step_days: step between rebalance dates
        min_coverage: minimum data coverage required
        transaction_cost_bps: transaction costs in basis points
        volume: DataFrame of daily volume, or None
        purge_days: gap between train and test windows
        return_weights: if True, also return weight history for turnover analysis
        oos_start: out-of-sample start date (str or Timestamp). If provided, only
                   backtest from this date onward (for neural methods trained on prior data).
        weight_ema: EMA smoothing factor for neural method weights (0=no smoothing,
                    0.3=blend 30% new + 70% old). Reduces turnover. Only applies to
                    DHRP, LLM_DHRP, MLP, Transformer, PPO.
        strict_methods: methods whose failures should raise immediately.
        return_diagnostics: if True, return fallback/failure counts with results.
    Returns:
        DataFrame with columns [method, date, return]
        If return_weights=True, returns (results_df, weights_history)
        where weights_history is a list of dicts with [method, date, weights]
    """
    if methods is None:
        # Only include methods that have models provided
        methods = ["EW", "MINVAR", "MV", "HRP", "RP", "MAXDIV"]
        if dhrp_model is not None:
            methods.append("DHRP")
        if llm_dhrp_model is not None:
            methods.append("LLM_DHRP")
        if mlp_model is not None:
            methods.append("MLP")
        if transformer_model is not None:
            methods.append("Transformer")
        if ppo_model is not None:
            methods.append("PPO")
        if dfl_model is not None:
            methods.append("DFL")

    # Per-universe overrides: lookback, weekly rebalance, covariance shrinkage.
    # When `universe` is None the function's own defaults / is_em branch are
    # used, so existing callers behave identically.
    if universe is not None:
        cfg = get_universe_config(universe)
        train_days = cfg.get("lookback_window", train_days)
        step_days = cfg.get("rebalance_freq", step_days)
        cov_shrinkage = cfg.get("cov_shrinkage", 0.001 if is_em else 1e-6)
    else:
        cov_shrinkage = 0.001 if is_em else 1e-6

    # dropna(how="all") keeps the full 10-year span even if one asset has a
    # late inception (e.g. CPER in Commodities). Per-asset NaNs are zeroed by
    # the .fillna(0) on the train/test slices below.
    rets = compute_returns(prices, how="all")
    results = []
    weights_history = []
    prev_weights = {}
    fallback_counts = defaultdict(int)
    failure_counts = defaultdict(int)
    failure_details = []
    strict_methods = set(strict_methods or [])
    neural_methods = {"DHRP", "LLM_DHRP", "MLP", "Transformer", "PPO", "DFL"}
    rebalance_idx = 0

    # Determine OOS start index
    oos_idx = train_days
    if oos_start is not None:
        oos_date = pd.Timestamp(oos_start)
        oos_candidates = rets.index[rets.index >= oos_date]
        if len(oos_candidates) > 0:
            oos_idx = max(train_days, rets.index.get_loc(oos_candidates[0]))

    for t in range(train_days, len(rets), step_days):
        if t < oos_idx:
            rebalance_idx += 1
            continue
        # Apply purge gap: train ends purge_days before test starts
        train_end = t - purge_days if purge_days > 0 else t
        if train_end <= train_days:
            rebalance_idx += 1
            continue
        # Compute per-asset coverage BEFORE imputation so late-inception assets
        # are correctly excluded. Using fillna(0) before the check would silence
        # the missingness signal entirely.
        train_raw = rets.iloc[train_end - train_days : train_end]
        # Drop assets whose coverage in this train window falls below
        # min_coverage (e.g. CPER prior to its inception in late 2011 / 2014).
        per_asset_coverage = train_raw.notna().mean()
        keep_assets = per_asset_coverage[per_asset_coverage >= min_coverage].index
        if len(keep_assets) < 2:
            rebalance_idx += 1
            continue
        train = train_raw[keep_assets].fillna(0)
        # Window-level coverage check (overall fraction of present cells).
        if train_raw[keep_assets].notna().mean().mean() < min_coverage:
            rebalance_idx += 1
            continue
        mu = train.mean().values * 252
        # Ledoit-Wolf shrinkage for better-conditioned covariance estimation
        try:
            cov = ledoit_wolf_cov(train)
        except Exception:
            cov = train.cov().values * 252
        cov = cov + np.eye(train.shape[1]) * cov_shrinkage
        # Record only until the next rebalance fires, so each OOS date
        # appears exactly once per method. Prevents duplicate-date
        # inflation of Sharpe / annualized return / cumulative plots
        # when step_days < test_days (e.g. Commodities: step=5, test=21).
        # When step_days >= test_days this is a no-op (preserves legacy
        # behavior for DM and EM).
        record_days = min(test_days, step_days)
        # Restrict test slice to the same assets we trained on so that
        # late-inception assets do not contribute synthetic zero returns.
        test = rets.iloc[t : t + record_days][keep_assets].fillna(0)
        if test.empty:
            rebalance_idx += 1
            continue

        # Volume window for this rebalance
        vol_window = None
        if volume is not None and not volume.empty:
            vol_window = volume.iloc[max(0, train_end - train_days) : train_end]

        for m in methods:
            try:
                w = _compute_weights(
                    m, mu, cov, train, is_em,
                    dhrp_model=dhrp_model,
                    llm_dhrp_model=llm_dhrp_model,
                    mlp_model=mlp_model,
                    transformer_model=transformer_model,
                    ppo_model=ppo_model,
                    dfl_model=dfl_model,
                    text_features=text_features,
                    macro_features=macro_features,
                    rebalance_idx=rebalance_idx,
                    rebalance_date=rets.index[t],
                    volume=vol_window,
                )
                if w is None:
                    continue
                w = np.asarray(w, dtype=float).reshape(-1)
                if len(w) != len(keep_assets):
                    raise ValueError(
                        f"{m} returned {len(w)} weights for {len(keep_assets)} assets"
                    )
                w_ser = pd.Series(w, index=keep_assets).replace([np.inf, -np.inf], np.nan)
                w_ser = w_ser.fillna(0.0).clip(lower=0.0)
                if w_ser.sum() <= 0:
                    fallback_counts[m] += 1
                    raise ValueError(f"{m} returned non-positive or all-zero weights")
                w_ser = w_ser / w_ser.sum()

                # EMA weight smoothing for neural methods to reduce turnover
                if weight_ema > 0 and m in neural_methods and m in prev_weights:
                    prev_current = prev_weights[m].reindex(w_ser.index, fill_value=0.0)
                    w_ser = (1 - weight_ema) * w_ser + weight_ema * prev_current
                    w_ser = w_ser.clip(lower=0.0)
                    if w_ser.sum() <= 0:
                        fallback_counts[m] += 1
                        raise ValueError(f"{m} EMA smoothing produced zero weights")
                    w_ser = w_ser / w_ser.sum()

                # Record weights for turnover analysis
                if return_weights:
                    weights_history.append({
                        "method": m,
                        "date": rets.index[t],
                        "assets": list(w_ser.index),
                        "weights": w_ser.values.tolist(),
                    })

                # Transaction cost adjustment
                tc_drag = 0.0
                if transaction_cost_bps > 0 and m in prev_weights:
                    union = w_ser.index.union(prev_weights[m].index)
                    turnover = np.abs(
                        w_ser.reindex(union, fill_value=0.0)
                        - prev_weights[m].reindex(union, fill_value=0.0)
                    ).sum()
                    tc_drag = turnover * transaction_cost_bps / 10000
                prev_weights[m] = w_ser.copy()

                daily_rs = (test[w_ser.index].values @ w_ser.values).flatten()
                n_days = len(daily_rs)
                for i, r in enumerate(daily_rs):
                    # Amortize tc drag over the days actually recorded,
                    # not the full test_days (which may exceed step_days).
                    adj_r = float(r) - (tc_drag / max(n_days, 1) if i == 0 else 0)
                    results.append({"method": m, "date": test.index[i], "return": adj_r})
            except Exception as exc:
                failure_counts[m] += 1
                failure_details.append({
                    "method": m,
                    "date": rets.index[t],
                    "error": f"{type(exc).__name__}: {exc}",
                })
                if m in strict_methods:
                    raise RuntimeError(f"rolling_backtest failed for strict method {m} at {rets.index[t]}") from exc
                warnings.warn(
                    f"rolling_backtest skipped {m} at {rets.index[t]}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        rebalance_idx += 1

    results_df = pd.DataFrame(results)
    diagnostics = {
        "fallback_counts": dict(fallback_counts),
        "failure_counts": dict(failure_counts),
        "failure_details": failure_details,
    }
    for method in strict_methods:
        if diagnostics["fallback_counts"].get(method, 0):
            raise AssertionError(f"{method} had fallback weight computations")
        if diagnostics["failure_counts"].get(method, 0):
            raise AssertionError(f"{method} had failed weight computations")

    if return_weights and return_diagnostics:
        return results_df, weights_history, diagnostics
    if return_weights:
        return results_df, weights_history
    if return_diagnostics:
        return results_df, diagnostics
    return results_df


def multiseed_backtest(prices, models_by_seed, is_em=False, methods=None,
                       rf=0.03, **kwargs):
    """Run backtests across multiple seeds and aggregate results.

    Args:
        prices: DataFrame of adjusted close prices
        models_by_seed: list of dicts, each mapping model_type to trained model
                        e.g. [{"dhrp": model_s0, "mlp": model_s0}, ...]
        is_em: emerging markets flag
        methods: list of method names
        rf: risk-free rate
        **kwargs: additional args passed to rolling_backtest
    Returns:
        DataFrame with columns [Method, Sharpe_mean, Sharpe_std, Sortino_mean, ...]
    """
    all_stats = []
    for seed_models in models_by_seed:
        res = rolling_backtest(
            prices, is_em=is_em,
            dhrp_model=seed_models.get("dhrp"),
            llm_dhrp_model=seed_models.get("llm_dhrp"),
            mlp_model=seed_models.get("mlp"),
            transformer_model=seed_models.get("transformer"),
            ppo_model=seed_models.get("ppo"),
            methods=methods,
            **kwargs,
        )
        stats = compute_stats(res, rf=rf)
        all_stats.append(stats)

    if not all_stats:
        return pd.DataFrame()

    # Aggregate across seeds
    combined = pd.concat(all_stats, ignore_index=True)
    metric_cols = [c for c in combined.columns if c != "Method"]
    agg = combined.groupby("Method")[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


def _compute_weights(
    method, mu, cov, train_rets, is_em,
    dhrp_model=None, llm_dhrp_model=None,
    mlp_model=None, transformer_model=None, ppo_model=None,
    dfl_model=None,
    text_features=None, macro_features=None,
    rebalance_idx=0, rebalance_date=None,
    volume=None,
):
    """Compute portfolio weights for a given method."""
    n_assets = train_rets.shape[1]

    def _check_model(method_name, model):
        if model is None:
            return
        if hasattr(model, "n_assets") and model.n_assets != n_assets:
            raise ValueError(f"{method_name} model expects {model.n_assets} assets, got {n_assets}")

    if method == "EW":
        return equal_weight(mu, cov)
    elif method == "MINVAR":
        return min_variance(mu, cov, is_em)
    elif method == "MV":
        return mean_variance(mu, cov, is_em)
    elif method == "HRP":
        return hrp_allocation(cov)
    elif method == "RP":
        return risk_parity(cov)
    elif method == "MAXDIV":
        return max_diversification(mu, cov)
    elif method == "DHRP" and dhrp_model is not None:
        _check_model(method, dhrp_model)
        return dhrp_weights(dhrp_model, train_rets, is_em, volume=volume)
    elif method == "LLM_DHRP" and llm_dhrp_model is not None:
        _check_model(method, llm_dhrp_model)
        return _llm_dhrp_weights(
            llm_dhrp_model, train_rets, is_em,
            text_features, macro_features,
            rebalance_idx, rebalance_date,
            volume=volume,
        )
    elif method == "MLP" and mlp_model is not None:
        _check_model(method, mlp_model)
        return dhrp_weights(mlp_model, train_rets, is_em, volume=volume)
    elif method == "Transformer" and transformer_model is not None:
        _check_model(method, transformer_model)
        return dhrp_weights(transformer_model, train_rets, is_em, volume=volume)
    elif method == "PPO" and ppo_model is not None:
        _check_model(method, ppo_model)
        return dhrp_weights(ppo_model, train_rets, is_em, volume=volume)
    elif method == "DFL" and dfl_model is not None:
        _check_model(method, dfl_model)
        return dhrp_weights(dfl_model, train_rets, is_em, volume=volume)
    return None


def _llm_dhrp_weights(
    model, rets, is_em,
    text_features, macro_features,
    rebalance_idx, rebalance_date,
    volume=None,
):
    """Compute LLM-DHRP portfolio weights."""
    import torch
    device = next(model.parameters()).device
    if hasattr(model, "n_assets") and model.n_assets != rets.shape[1]:
        raise ValueError(f"LLM_DHRP model expects {model.n_assets} assets, got {rets.shape[1]}")
    cov = (rets.cov().values * 252).astype(np.float32)
    if is_em:
        cov += np.eye(cov.shape[0]) * 0.01
    feat = make_features(rets, model.feature_dim, is_em, volume=volume).astype(np.float32)

    text_emb = None
    if text_features is not None and model.use_text:
        fb = text_features.get("finbert") if isinstance(text_features, dict) else None
        if fb is not None:
            emb = None
            if isinstance(fb, dict):
                if rebalance_date is not None:
                    ts = pd.Timestamp(rebalance_date)
                    prior = [k for k in fb.keys() if pd.Timestamp(k) <= ts]
                    if prior:
                        emb = fb[max(prior)]
            elif rebalance_idx < len(fb):
                emb = fb[rebalance_idx]
            if emb is not None:
                emb = np.asarray(emb)
                if emb.ndim == 2:
                    emb = aggregate_text_per_timestep(
                        emb[None, ...], method="norm_mean_max_concat"
                    )[0]
                text_emb = torch.from_numpy(emb.astype(np.float32)).to(device)

    macro_feat = None
    if macro_features is not None and model.use_macro and rebalance_date is not None:
        macro_feat = get_macro_vector(macro_features, rebalance_date)
        macro_feat = torch.from_numpy(macro_feat).to(device)

    with torch.no_grad():
        w = model(
            torch.from_numpy(np.nan_to_num(feat)).to(device),
            torch.from_numpy(np.nan_to_num(cov)).to(device),
            text_emb=text_emb,
            macro_feat=macro_feat,
        ).cpu().numpy()

    w = np.nan_to_num(np.clip(w, 0, 1))
    if w.sum() <= 0:
        raise ValueError("LLM_DHRP returned non-positive weight sum")
    return w / w.sum()

# ====================================================================
# Module: factor_analysis.py
# ====================================================================
import os

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")


def load_ff5_factors(start, end):
    """Load Fama-French 5 factors + Momentum from Ken French's data library.

    Returns DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF
    """
    import pandas_datareader.data as pdr

    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"FF5Mom_{start}_{end}.csv")
    if os.path.exists(cp):
        ff = pd.read_csv(cp, index_col=0, parse_dates=True)
        print(f"  FF5+Mom factors: {ff.shape[0]} days (cached)")
        return ff

    # FF5 daily factors
    try:
        ff5 = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start, end)[0] / 100
    except Exception:
        ff5 = pdr.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start, end)[0] / 100

    # Momentum factor
    try:
        mom = pdr.DataReader("F-F_Momentum_Factor_daily", "famafrench", start, end)[0] / 100
        mom.columns = ["Mom"]
    except Exception:
        mom = pd.DataFrame(index=ff5.index, columns=["Mom"], data=0.0)

    ff = pd.concat([ff5, mom], axis=1, join="inner")
    ff.to_csv(cp)
    print(f"  FF5+Mom factors: {ff.shape[0]} days")
    return ff


def load_aqr_commodity_factors(start, end):
    """Load AQR Value and Momentum Everywhere commodity factors (monthly).

    Returns DataFrame with columns: VAL_CM, MOM_CM
    Source: Asness, Moskowitz, Pedersen (2013), updated by AQR.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"AQR_CommodityFactors_{start}_{end}.csv")
    if os.path.exists(cp):
        df = pd.read_csv(cp, index_col=0, parse_dates=True)
        print(f"  AQR Commodity factors: {df.shape[0]} months (cached)")
        return df

    url = (
        "https://images.aqr.com/-/media/AQR/Documents/Insights/"
        "Data-Sets/Value-and-Momentum-Everywhere-Factors-Monthly.xlsx"
    )
    raw = pd.read_excel(url, sheet_name="VME Factors", header=None)

    # Row 21 has column names, data starts at row 22
    headers = raw.iloc[21].tolist()
    data = raw.iloc[22:].copy()
    data.columns = headers
    data = data.rename(columns={"DATE": "date"})
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date")

    # Extract commodity value and momentum columns
    cm_cols = {"VALLS_VME_COM": "VAL_CM", "MOMLS_VME_COM": "MOM_CM"}
    df = data[[c for c in cm_cols if c in data.columns]].rename(columns=cm_cols)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.loc[start:end]
    df.to_csv(cp)
    print(f"  AQR Commodity factors: {df.shape[0]} months")
    return df


def factor_analysis(results, ff, methods=None):
    """Run Fama-French factor regressions for each method.

    Automatically uses FF5+Mom if available, falls back to FF3.
    """
    if methods is None:
        methods = sorted(results["method"].unique())

    # Determine available factors
    available = ff.columns.tolist()
    core_factors = ["Mkt-RF", "SMB", "HML"]
    extra_factors = [f for f in ["RMW", "CMA", "Mom"] if f in available]
    all_factors = core_factors + extra_factors
    factor_cols = [f for f in all_factors if f in available]
    n_factors = len(factor_cols)

    stats = []
    for m in methods:
        ser = results[results["method"] == m].set_index("date")["return"]
        ser.index = pd.to_datetime(ser.index)
        merged = pd.concat([ser.rename("ret"), ff], axis=1, join="inner").dropna()
        if len(merged) < 100:
            continue

        X = merged[factor_cols].values
        y = merged["ret"].values
        mod = OLS(y, add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

        row = {
            "Method": m,
            "Alpha_ann": mod.params[0] * 252,
            "Alpha_t": mod.tvalues[0],
            "Alpha_p": mod.pvalues[0],
            "R2_adj": mod.rsquared_adj,
        }
        for i, f in enumerate(factor_cols):
            row[f"Beta_{f}"] = mod.params[i + 1]
            row[f"t_{f}"] = mod.tvalues[i + 1]

        stats.append(row)
    return pd.DataFrame(stats)


def commodity_factor_analysis(results, aqr_factors, methods=None):
    """Run commodity-specific factor regressions using AQR Value & Momentum.

    AQR factors are monthly; daily returns are aggregated to monthly before
    regression. Factors: VAL_CM (commodity value), MOM_CM (commodity momentum).
    """
    if methods is None:
        methods = sorted(results["method"].unique())

    factor_cols = [c for c in ["VAL_CM", "MOM_CM"] if c in aqr_factors.columns]
    if not factor_cols:
        raise ValueError("AQR commodity factors not found")

    stats = []
    for m in methods:
        ser = results[results["method"] == m].set_index("date")["return"]
        ser.index = pd.to_datetime(ser.index)
        # Aggregate daily returns to monthly
        monthly = (1 + ser).resample("ME").prod() - 1
        merged = pd.concat(
            [monthly.rename("ret"), aqr_factors[factor_cols]], axis=1, join="inner"
        ).dropna()
        if len(merged) < 24:
            continue

        X = merged[factor_cols].values
        y = merged["ret"].values
        mod = OLS(y, add_constant(X)).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4}
        )

        row = {
            "Method": m,
            "Alpha_ann": mod.params[0] * 12,
            "Alpha_t": mod.tvalues[0],
            "Alpha_p": mod.pvalues[0],
            "R2_adj": mod.rsquared_adj,
        }
        for i, f in enumerate(factor_cols):
            row[f"Beta_{f}"] = mod.params[i + 1]
            row[f"t_{f}"] = mod.tvalues[i + 1]

        stats.append(row)
    return pd.DataFrame(stats)

# ====================================================================
# Module: statistics.py
# ====================================================================
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_rel
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac


def compute_sharpe(r, rf=0.03):
    """Compute annualized Sharpe ratio."""
    exc = r - rf / 252
    return (exc.mean() * 252) / (exc.std() * np.sqrt(252)) if exc.std() > 0 else 0


def probabilistic_sharpe_ratio(r, sr_threshold=0.0, rf=0.03):
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012, Two Sigma 2017).

    Probability that the true Sharpe ratio exceeds `sr_threshold`, accounting for
    sample size, skewness, and kurtosis. PSR > 0.95 means significantly positive
    Sharpe at 95% confidence — gold-standard significance test in quant finance
    and increasingly required by NeurIPS/ICLR finance reviewers.

    Args:
        r: array of daily returns (not annualized)
        sr_threshold: annualized Sharpe ratio benchmark (default 0)
        rf: annualized risk-free rate (default 3%)
    Returns:
        PSR in [0, 1] (higher = more confident the true Sharpe exceeds threshold)
    """
    n = len(r)
    if n < 30:
        return np.nan

    exc = r - rf / 252
    if exc.std() <= 0:
        return np.nan

    # Daily Sharpe (not annualized) for the PSR formula
    sr_hat_daily = exc.mean() / exc.std()

    # Daily threshold (convert annualized SR threshold)
    sr_thresh_daily = sr_threshold / np.sqrt(252)

    # Higher moments of excess returns (centered)
    centered = exc - exc.mean()
    sigma = exc.std()
    skew = (centered ** 3).mean() / (sigma ** 3) if sigma > 0 else 0.0
    kurt = (centered ** 4).mean() / (sigma ** 4) if sigma > 0 else 3.0  # raw kurtosis

    # Standard error of Sharpe (Mertens 2002 / Bailey-LdP correction)
    denom = 1 - skew * sr_hat_daily + ((kurt - 1) / 4) * (sr_hat_daily ** 2)
    if denom <= 0:
        return np.nan
    se = np.sqrt(denom / (n - 1))

    z = (sr_hat_daily - sr_thresh_daily) / se
    return float(norm.cdf(z))


def compute_stats(results, rf=0.03, n_boot=1000, gamma=2.5, oos_start=None):
    """Compute comprehensive performance metrics for each method.

    If ``oos_start`` is provided, restrict metrics to dates >= oos_start so
    headline numbers reflect strict out-of-sample performance.

    Returns DataFrame with columns: Method, Sharpe, Sortino, Calmar, MaxDD,
    CVaR_5, VaR_5, Omega, CER, Ann_Return, Ann_Vol, HAC_t, CI_lo, CI_hi.
    """
    if oos_start is not None and "date" in results.columns and not results.empty:
        cutoff = pd.Timestamp(oos_start)
        results = results[pd.to_datetime(results["date"]) >= cutoff]
    # Defensive dedup: if rolling_backtest ever emits duplicate (method, date)
    # rows (overlapping rebalance windows), average them so Sharpe /
    # Ann_Return / MaxDD / cumulative plots aren't multiply-counted.
    if "date" in results.columns and not results.empty:
        dup_mask = results.duplicated(subset=["method", "date"], keep=False)
        if dup_mask.any():
            import warnings
            n_dup = int(dup_mask.sum())
            warnings.warn(
                f"compute_stats: {n_dup} duplicate (method, date) rows detected; "
                "averaging. This usually means rolling_backtest was run with "
                "step_days < test_days. Expected to be zero after the "
                "backtest.py fix — investigate if you see this.",
                stacklevel=2,
            )
            results = (
                results.groupby(["method", "date"], as_index=False)["return"].mean()
            )

    stats = []
    for m in sorted(results["method"].unique()):
        r = (
            results[results["method"] == m]
            .sort_values("date")["return"] if "date" in results.columns
            else results[results["method"] == m]["return"]
        ).dropna().values
        if len(r) == 0:
            continue
        exc = r - rf / 252
        vol = exc.std()
        ann_ret = exc.mean() * 252
        ann_vol = vol * np.sqrt(252) if vol > 0 else np.nan
        sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan

        # HAC t-stat
        try:
            if len(exc) >= 30:
                mod = OLS(exc, add_constant(np.zeros(len(exc)))).fit()
                hac_t = (exc.mean() * 252) / np.sqrt(cov_hac(mod)[0, 0] * 252)
            else:
                hac_t = np.nan
        except Exception:
            hac_t = np.nan

        # BCa bootstrap CI
        ci_lo, ci_hi = _bca_bootstrap_ci(r, rf=rf, n_boot=n_boot)

        # Max drawdown
        cum = (1 + pd.Series(r)).cumprod()
        peak = cum.cummax()
        mdd = ((cum - peak) / peak).min()

        # Calmar ratio
        calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-8 else np.nan

        # Sortino ratio (downside deviation)
        downside = exc[exc < 0]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 1 else np.nan
        sortino = ann_ret / downside_std if downside_std and downside_std > 0 else np.nan

        # VaR and CVaR at 5%
        var_5 = float(np.percentile(r, 5))
        cvar_5 = float(r[r <= var_5].mean()) if np.any(r <= var_5) else var_5

        # Omega ratio (threshold = 0)
        gains = r[r > 0].sum()
        losses = abs(r[r <= 0].sum())
        omega = gains / losses if losses > 1e-12 else np.nan

        # Certainty Equivalent Return (CER) under CRRA utility
        try:
            u_vals = ((1 + np.clip(r, -0.99, None)) ** (1 - gamma)) / (1 - gamma)
            eu = u_vals.mean()
            cer = (eu * (1 - gamma)) ** (1 / (1 - gamma)) - 1
            cer_ann = cer * 252
        except Exception:
            cer_ann = np.nan

        # Probabilistic Sharpe Ratio (gold-standard finance significance test)
        psr = probabilistic_sharpe_ratio(r, sr_threshold=0.0, rf=rf)

        stats.append({
            "Method": m, "Sharpe": sharpe, "Sortino": sortino,
            "Calmar": calmar, "MaxDD": mdd,
            "CVaR_5": cvar_5, "VaR_5": var_5, "Omega": omega,
            "CER": cer_ann, "Ann_Return": ann_ret, "Ann_Vol": ann_vol,
            "HAC_t": hac_t, "CI_lo": ci_lo, "CI_hi": ci_hi,
            "PSR": psr,
        })
    return pd.DataFrame(stats)


def _bca_bootstrap_ci(r, rf=0.03, n_boot=1000, alpha=0.05):
    """BCa (bias-corrected accelerated) bootstrap confidence interval for Sharpe."""
    n = len(r)
    if n < 30:
        return np.nan, np.nan

    observed = compute_sharpe(r, rf)

    # Bootstrap distribution
    np.random.seed(42)
    boot_sharpes = []
    bl = max(5, int(n ** (1 / 3)))  # block length for block bootstrap
    for _ in range(n_boot):
        idx = np.random.randint(0, n - bl + 1, n // bl + 1)
        br = np.concatenate([r[i : i + bl] for i in idx])[:n]
        exc = br - rf / 252
        if exc.std() > 0:
            boot_sharpes.append((exc.mean() * 252) / (exc.std() * np.sqrt(252)))
    if len(boot_sharpes) < 50:
        return np.nan, np.nan
    boot_sharpes = np.array(boot_sharpes)

    # Bias correction factor (z0)
    z0 = norm.ppf(np.mean(boot_sharpes < observed))

    # Acceleration factor (a) via jackknife
    jk_sharpes = np.zeros(n)
    for i in range(n):
        ri = np.delete(r, i)
        exc = ri - rf / 252
        jk_sharpes[i] = (exc.mean() * 252) / (exc.std() * np.sqrt(252)) if exc.std() > 0 else 0
    jk_mean = jk_sharpes.mean()
    num = np.sum((jk_mean - jk_sharpes) ** 3)
    den = 6 * (np.sum((jk_mean - jk_sharpes) ** 2) ** 1.5)
    a = num / den if abs(den) > 1e-12 else 0

    # Adjusted percentiles
    z_alpha = norm.ppf(alpha / 2)
    z_1alpha = norm.ppf(1 - alpha / 2)

    p_lo = norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    p_hi = norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    p_lo = np.clip(p_lo, 0.5 / n_boot, 1 - 0.5 / n_boot)
    p_hi = np.clip(p_hi, 0.5 / n_boot, 1 - 0.5 / n_boot)

    return float(np.percentile(boot_sharpes, p_lo * 100)), float(np.percentile(boot_sharpes, p_hi * 100))


def jobson_korkie_test(r_a, r_b, rf=0.03):
    """Jobson-Korkie (1981) test for Sharpe ratio equality with
    Memmel (2003) correction. Returns (z_stat, p_value).

    More powerful than naive bootstrap for moderate-N samples.
    """
    n = len(r_a)
    if n < 30:
        return np.nan, np.nan
    exc_a = r_a - rf / 252
    exc_b = r_b - rf / 252
    mu_a, mu_b = exc_a.mean(), exc_b.mean()
    sd_a, sd_b = exc_a.std(ddof=1), exc_b.std(ddof=1)
    if sd_a < 1e-12 or sd_b < 1e-12:
        return np.nan, np.nan
    sr_a, sr_b = mu_a / sd_a, mu_b / sd_b
    rho = np.corrcoef(exc_a, exc_b)[0, 1]
    # Memmel-corrected variance
    var = (1.0 / n) * (
        2 * (1 - rho)
        + 0.5 * (sr_a ** 2 + sr_b ** 2 - 2 * sr_a * sr_b * rho ** 2)
    )
    if var <= 0:
        return np.nan, np.nan
    z = (sr_a - sr_b) / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


def _stationary_bootstrap_indices(n, avg_block_len, rng):
    p = 1.0 / max(float(avg_block_len), 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t_idx in range(1, n):
        if rng.random() < p:
            idx[t_idx] = rng.integers(0, n)
        else:
            idx[t_idx] = (idx[t_idx - 1] + 1) % n
    return idx


def sharpe_difference_test(results, method_a="DHRP", method_b="HRP", n_boot=1000,
                           rf=0.03, avg_block_len=None, seed=42):
    """Paired stationary-bootstrap test for a Sharpe-ratio difference."""
    ser_a = (
        results[results["method"] == method_a]
        .groupby("date")["return"].mean()
        .rename("A")
    )
    ser_b = (
        results[results["method"] == method_b]
        .groupby("date")["return"].mean()
        .rename("B")
    )
    merged = pd.concat([ser_a, ser_b], axis=1).dropna().sort_index()
    if len(merged) < 30:
        raise ValueError(f"Need >=30 paired dates; got {len(merged)}")

    r_a, r_b = merged["A"].values, merged["B"].values
    n = len(r_a)
    if avg_block_len is None:
        avg_block_len = max(5, int(round(n ** (1 / 3))))

    sr_a = compute_sharpe(r_a, rf)
    sr_b = compute_sharpe(r_b, rf)
    actual_diff = sr_a - sr_b
    rng = np.random.default_rng(seed)

    sharpe_diffs = np.empty(n_boot, dtype=float)
    for j in range(n_boot):
        idx = _stationary_bootstrap_indices(n, avg_block_len, rng)
        sharpe_diffs[j] = compute_sharpe(r_a[idx], rf) - compute_sharpe(r_b[idx], rf)

    ci_lo, ci_hi = np.percentile(sharpe_diffs, [2.5, 97.5])
    p_boot = 2 * min(np.mean(sharpe_diffs <= 0.0), np.mean(sharpe_diffs >= 0.0))
    p_boot = float(min(max(p_boot, 1.0 / n_boot), 1.0))

    diff_returns = r_a - r_b
    diff_std = diff_returns.std(ddof=1)
    ir = (
        diff_returns.mean() * 252 / (diff_std * np.sqrt(252))
        if diff_std > 0 else np.nan
    )
    cohens_d = diff_returns.mean() / (diff_std + 1e-12)

    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_dates": n,
        "sharpe_a": sr_a,
        "sharpe_b": sr_b,
        "sharpe_diff": actual_diff,
        "bootstrap_ci_lo": float(ci_lo),
        "bootstrap_ci_hi": float(ci_hi),
        "bootstrap_se": float(sharpe_diffs.std(ddof=1)),
        "bootstrap_p": p_boot,
        "bootstrap_p_approx": p_boot,
        "primary_p": p_boot,
        "avg_block_len": avg_block_len,
        "information_ratio": float(ir),
        "cohens_d": float(cohens_d),
    }


def fdr_correct(p_values, method="bh"):
    """Multiple testing correction.

    Args:
        p_values: list or array of raw p-values
        method: "bh" for Benjamini-Hochberg, "bonferroni", or "holm" for Holm-Bonferroni
    Returns:
        dict with 'raw', 'adjusted', and 'significant' arrays
    """
    p = np.array(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return {"raw": p, "adjusted": p, "significant_005": np.array([], dtype=bool),
                "significant_010": np.array([], dtype=bool)}

    if method == "bonferroni":
        adjusted = np.minimum(p * n, 1.0)
    elif method == "holm":
        # Holm-Bonferroni: step-down procedure controlling FWER
        sorted_idx = np.argsort(p)
        adjusted = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            adjusted[idx] = p[idx] * (n - i)
        # Enforce monotonicity (step-down)
        running_max = 0.0
        for i in range(n):
            idx = sorted_idx[i]
            adjusted[idx] = max(adjusted[idx], running_max)
            running_max = adjusted[idx]
        adjusted = np.minimum(adjusted, 1.0)
    else:  # Benjamini-Hochberg
        sorted_idx = np.argsort(p)
        sorted_p = p[sorted_idx]
        adjusted = np.zeros(n)
        for i in range(n):
            adjusted[sorted_idx[i]] = sorted_p[i] * n / (i + 1)
        # Enforce monotonicity
        for i in range(n - 2, -1, -1):
            adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i]], adjusted[sorted_idx[i + 1]] if i + 1 < n else 1.0)
        adjusted = np.minimum(adjusted, 1.0)

    return {
        "raw": p,
        "adjusted": adjusted,
        "significant_005": adjusted < 0.05,
        "significant_010": adjusted < 0.10,
    }


def diebold_mariano_test(results, method_a="DHRP", method_b="HRP", loss_fn="squared"):
    """Diebold-Mariano test for equal predictive accuracy."""
    ser_a = results[results["method"] == method_a].set_index("date")["return"]
    ser_b = results[results["method"] == method_b].set_index("date")["return"]
    merged = pd.concat([ser_a.rename("A"), ser_b.rename("B")], axis=1).dropna()

    if loss_fn == "squared":
        # Use squared returns as loss (lower = better risk management)
        e_a = merged["A"].values ** 2
        e_b = merged["B"].values ** 2
    else:
        # Use negative returns as loss (higher return = lower loss)
        e_a = -merged["A"].values
        e_b = -merged["B"].values

    d = e_a - e_b
    T = len(d)
    d_mean = d.mean()

    # Newey-West variance with 1 lag
    gamma_0 = np.var(d, ddof=1)
    if T > 1:
        gamma_1 = np.cov(d[1:], d[:-1])[0, 1]
        V = gamma_0 + 2 * gamma_1
    else:
        V = gamma_0
    V = max(V, 1e-12)

    DM = d_mean / np.sqrt(V / T)
    p_value = 2 * (1 - norm.cdf(abs(DM)))
    return {"DM_stat": DM, "p_value": p_value, "loss_fn": loss_fn}


# ---------------------------------------------------------------------------
# Pairwise statistical tests with multiple testing correction
# ---------------------------------------------------------------------------

def pairwise_sharpe_tests(results, ref="DHRP", n_boot=1000, rf=0.03, correction="holm"):
    """Run pairwise Sharpe difference tests between ref and all other methods.

    Uses paired stationary-bootstrap Sharpe differences as the declared
    primary test, with Holm-Bonferroni correction by default.
    """
    methods = sorted(results["method"].unique())
    if ref not in methods:
        return pd.DataFrame()
    others = [m for m in methods if m != ref]
    rows = []
    raw_ps = []
    for other in others:
        try:
            res = sharpe_difference_test(results, method_a=ref, method_b=other,
                                         n_boot=n_boot, rf=rf)
            rows.append(res)
            raw_ps.append(res.get("primary_p", res["bootstrap_p"]))
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    corrected = fdr_correct(raw_ps, method=correction)
    df = pd.DataFrame(rows)
    df["adjusted_p"] = corrected["adjusted"]
    df["significant_005"] = corrected["significant_005"]
    df["significant_010"] = corrected["significant_010"]
    return df


def pairwise_dm_tests(results, ref="DHRP", loss_fn="squared", correction="holm"):
    """Run pairwise Diebold-Mariano tests between ref and all other methods.

    Returns DataFrame with raw and corrected p-values.
    """
    methods = sorted(results["method"].unique())
    if ref not in methods:
        return pd.DataFrame()
    others = [m for m in methods if m != ref]
    rows = []
    raw_ps = []
    for other in others:
        try:
            res = diebold_mariano_test(results, method_a=ref, method_b=other,
                                       loss_fn=loss_fn)
            res["method_a"] = ref
            res["method_b"] = other
            rows.append(res)
            raw_ps.append(res["p_value"])
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    corrected = fdr_correct(raw_ps, method=correction)
    df = pd.DataFrame(rows)
    df["adjusted_p"] = corrected["adjusted"]
    df["significant_005"] = corrected["significant_005"]
    return df


def full_statistical_battery(results, ref="DHRP", rf=0.03, n_boot=1000):
    """Run comprehensive statistical tests. Returns dict of DataFrames."""
    return {
        "sharpe_tests": pairwise_sharpe_tests(results, ref=ref, n_boot=n_boot, rf=rf),
        "dm_tests_squared": pairwise_dm_tests(results, ref=ref, loss_fn="squared"),
        "dm_tests_negative": pairwise_dm_tests(results, ref=ref, loss_fn="negative"),
        "spa": spa_test(results, benchmark="EW", n_boot=n_boot, rf=rf),
        "mcs": model_confidence_set(results, alpha=0.05, n_boot=n_boot, rf=rf),
    }


# ---------------------------------------------------------------------------
# Regime / sub-period analysis
# ---------------------------------------------------------------------------

def subperiod_analysis(results, rf=0.03, gamma=2.5, train_end=None):
    """Compute comprehensive metrics per macro regime.

    When ``train_end`` is provided, always emit an "OOS Full" row covering
    ``train_end`` through the end of the data so the OOS window is visible
    even if none of the legacy regime buckets line up with the split.

    Returns DataFrame with columns [Method, Period, Sharpe, Sortino, MaxDD, CVaR_5, N].
    """
    periods = {
        "Pre-COVID Bull (2016-2019)": ("2016-01-01", "2019-12-31"),
        "COVID Crash (2020-H1)": ("2020-01-01", "2020-06-30"),
        "Recovery (2020-H2 to 2021)": ("2020-07-01", "2021-12-31"),
        "Rate Hikes (2022 to 2023-H1)": ("2022-01-01", "2023-06-30"),
        "Post-Hike (2023-H2+)": ("2023-07-01", "2099-12-31"),
    }
    if train_end is not None:
        periods[f"OOS Full ({pd.Timestamp(train_end).strftime('%Y-%m-%d')}+)"] = (
            pd.Timestamp(train_end).strftime("%Y-%m-%d"), "2099-12-31",
        )
    rows = []
    for m in sorted(results["method"].unique()):
        ser = results[results["method"] == m].set_index("date")["return"]
        ser.index = pd.to_datetime(ser.index)
        for pname, (pstart, pend) in periods.items():
            sub = ser[(ser.index >= pstart) & (ser.index <= pend)]
            if len(sub) >= 20:
                r = sub.values
                exc = r - rf / 252
                vol = exc.std()
                ann_ret = exc.mean() * 252
                ann_vol = vol * np.sqrt(252) if vol > 0 else np.nan
                sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan

                # Sortino
                down = exc[exc < 0]
                down_std = down.std() * np.sqrt(252) if len(down) > 1 else np.nan
                sortino = ann_ret / down_std if down_std and down_std > 0 else np.nan

                # MaxDD
                cum = (1 + pd.Series(r)).cumprod()
                mdd = ((cum - cum.cummax()) / cum.cummax()).min()

                # CVaR
                var5 = np.percentile(r, 5)
                cvar5 = float(r[r <= var5].mean()) if np.any(r <= var5) else var5

                rows.append({
                    "Method": m, "Period": pname, "Sharpe": sharpe,
                    "Sortino": sortino, "MaxDD": mdd, "CVaR_5": cvar5,
                    "N": len(sub),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Transaction cost sensitivity
# ---------------------------------------------------------------------------

def compute_turnover(weights_history):
    """Compute average turnover from weight history.

    Args:
        weights_history: list of dicts with keys [method, date, weights]
    Returns:
        DataFrame with [Method, Avg_Turnover, Total_Turnover, N_Rebalances]
    """
    if not weights_history:
        return pd.DataFrame()
    df = pd.DataFrame(weights_history)
    rows = []
    for m in sorted(df["method"].unique()):
        mw = df[df["method"] == m].sort_values("date")
        w_arr = np.array(mw["weights"].tolist())
        if len(w_arr) < 2:
            continue
        turnovers = np.sum(np.abs(np.diff(w_arr, axis=0)), axis=1)
        rows.append({
            "Method": m,
            "Avg_Turnover": turnovers.mean(),
            "Total_Turnover": turnovers.sum(),
            "N_Rebalances": len(turnovers),
        })
    return pd.DataFrame(rows)


def cost_sensitivity_analysis(results, weights_history, cost_levels=None, rf=0.03):
    """Compute net-of-cost Sharpe at various transaction cost levels.

    Args:
        results: backtest results DataFrame
        weights_history: list of dicts with [method, date, weights]
        cost_levels: list of costs in bps (default [0, 5, 10, 20, 50])
    Returns:
        DataFrame with columns [Method, cost_0bps, cost_5bps, ...]
    """
    if cost_levels is None:
        cost_levels = [0, 5, 10, 20, 50]
    if not weights_history:
        return pd.DataFrame()

    wh_df = pd.DataFrame(weights_history)
    methods = sorted(results["method"].unique())
    rows = []

    for m in methods:
        r = results[results["method"] == m].sort_values("date")
        mw = wh_df[wh_df["method"] == m].sort_values("date")
        if mw.empty or r.empty:
            continue

        base_returns = r["return"].values.copy()
        w_arr = np.array(mw["weights"].tolist())

        # Compute turnover per rebalance
        turnovers = np.zeros(len(w_arr))
        if len(w_arr) > 1:
            turnovers[1:] = np.sum(np.abs(np.diff(w_arr, axis=0)), axis=1)

        # Map rebalance turnovers to daily returns
        rebal_dates = mw["date"].values
        daily_dates = r["date"].values

        row = {"Method": m}
        for cost_bps in cost_levels:
            adj_returns = base_returns.copy()
            # Spread turnover cost across holding period days
            for i, rd in enumerate(rebal_dates):
                mask = daily_dates == rd
                if mask.any():
                    tc = turnovers[i] * cost_bps / 10000
                    idx = np.where(mask)[0][0]
                    adj_returns[idx] -= tc
            exc = adj_returns - rf / 252
            vol = exc.std()
            s = (exc.mean() * 252) / (vol * np.sqrt(252)) if vol > 0 else np.nan
            row[f"cost_{cost_bps}bps"] = s
        rows.append(row)

    return pd.DataFrame(rows)


def breakeven_cost(results, weights_history, method_a="DHRP", method_b="HRP",
                   rf=0.03, max_bps=200):
    """Find the transaction cost level at which method_a's Sharpe equals method_b's."""
    for bps in range(0, max_bps + 1):
        df = cost_sensitivity_analysis(results, weights_history,
                                       cost_levels=[bps], rf=rf)
        if df.empty:
            return np.nan
        sa = df[df["Method"] == method_a][f"cost_{bps}bps"]
        sb = df[df["Method"] == method_b][f"cost_{bps}bps"]
        if sa.empty or sb.empty:
            return np.nan
        if sa.values[0] <= sb.values[0]:
            return bps
    return max_bps


# ---------------------------------------------------------------------------
# Computational benchmarks
# ---------------------------------------------------------------------------

def spa_test(results, benchmark="EW", n_boot=1000, rf=0.03):
    """Hansen's Superior Predictive Ability (SPA) test.

    Tests H0: no method is significantly better than the benchmark.
    Returns dict with SPA p-value and best method.
    """
    methods = sorted(results["method"].unique())
    if benchmark not in methods:
        return {"spa_pvalue": np.nan, "best_method": None}
    others = [m for m in methods if m != benchmark]

    bench_r = results[results["method"] == benchmark].set_index("date")["return"]
    d_matrix = []
    for m in others:
        m_r = results[results["method"] == m].set_index("date")["return"]
        merged = pd.concat([m_r.rename("m"), bench_r.rename("b")], axis=1).dropna()
        d = merged["m"].values - merged["b"].values
        d_matrix.append(d)

    # Align lengths
    min_len = min(len(d) for d in d_matrix)
    d_matrix = np.column_stack([d[:min_len] for d in d_matrix])
    T = d_matrix.shape[0]
    d_means = d_matrix.mean(axis=0)

    # Stationary bootstrap
    np.random.seed(42)
    max_t_star = np.zeros(n_boot)
    for b in range(n_boot):
        idx = np.random.choice(T, T, replace=True)
        boot_means = d_matrix[idx].mean(axis=0) - d_means
        boot_stds = d_matrix[idx].std(axis=0) + 1e-12
        max_t_star[b] = np.max(boot_means / boot_stds * np.sqrt(T))

    # Observed test statistic
    obs_stds = d_matrix.std(axis=0) + 1e-12
    obs_t = np.max(d_means / obs_stds * np.sqrt(T))
    spa_p = np.mean(max_t_star >= obs_t)
    best_idx = np.argmax(d_means)

    return {
        "spa_pvalue": spa_p,
        "best_method": others[best_idx],
        "best_excess_sharpe": compute_sharpe(
            results[results["method"] == others[best_idx]]["return"].values, rf
        ) - compute_sharpe(bench_r.values, rf),
    }


def model_confidence_set(results, alpha=0.05, n_boot=1000, rf=0.03):
    """Model Confidence Set (Hansen, Lunde, Nason 2011).

    Returns the set of models that cannot be distinguished from the best
    at significance level alpha.
    """
    methods = sorted(results["method"].unique())
    if len(methods) < 2:
        return methods

    # Compute loss matrix (negative Sharpe as loss)
    sharpes = {}
    for m in methods:
        r = results[results["method"] == m]["return"].dropna().values
        sharpes[m] = compute_sharpe(r, rf)

    # Start with all models
    mcs = list(methods)
    eliminated = True

    while eliminated and len(mcs) > 1:
        eliminated = False
        worst_method = None
        worst_p = 1.0

        for m in mcs:
            others_sharpe = np.mean([sharpes[o] for o in mcs if o != m])
            d = sharpes[m] - others_sharpe
            if d < 0:
                # Bootstrap test for elimination
                m_r = results[results["method"] == m]["return"].dropna().values
                all_r = {o: results[results["method"] == o]["return"].dropna().values
                         for o in mcs if o != m}
                np.random.seed(42)
                boot_count = 0
                n = min(len(m_r), min(len(v) for v in all_r.values()))
                for _ in range(n_boot):
                    idx = np.random.choice(n, n, replace=True)
                    boot_m = compute_sharpe(m_r[idx], rf)
                    boot_others = np.mean([compute_sharpe(v[idx], rf) for v in all_r.values()])
                    if boot_m - boot_others >= 0:
                        boot_count += 1
                p_val = boot_count / n_boot
                if p_val < worst_p:
                    worst_p = p_val
                    worst_method = m

        if worst_method is not None and worst_p < alpha:
            mcs.remove(worst_method)
            eliminated = True

    return mcs


def benchmark_efficiency(models_dict, n_assets=5, feature_dim=48, n_runs=100):
    """Benchmark inference time and parameter count for each model.

    Args:
        models_dict: dict mapping method name to (model, is_torch) tuple
        n_assets: number of assets for synthetic input
        feature_dim: feature dimension for synthetic input
        n_runs: number of forward passes to time
    Returns:
        DataFrame with [Method, Params, Inference_ms]
    """
    import torch

    rows = []

    for name, (model, is_torch) in models_dict.items():
        if is_torch:
            # Place synthetic inputs on same device as model
            dev = next(model.parameters()).device
            x_t = torch.randn(feature_dim, device=dev)
            S_t = torch.eye(n_assets, device=dev) * 0.04
            n_params = sum(p.numel() for p in model.parameters())
            model.eval()
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    model(x_t, S_t)
            # Timed runs
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_runs):
                    model(x_t, S_t)
            elapsed = (time.perf_counter() - t0) / n_runs * 1000  # ms
        else:
            n_params = 0
            t0 = time.perf_counter()
            for _ in range(n_runs):
                model(np.zeros(n_assets), np.eye(n_assets) * 0.04)
            elapsed = (time.perf_counter() - t0) / n_runs * 1000

        rows.append({"Method": name, "Params": n_params, "Inference_ms": elapsed})
    return pd.DataFrame(rows)

