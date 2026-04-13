"""Rolling-window backtest engine supporting all portfolio methods.

Supports:
- Classical baselines: EW, MINVAR, MV, HRP, RP, MAXDIV
- DHRP (differentiable HRP)
- LLM-DHRP (LLM-enhanced differentiable HRP)
- Deep baselines: MLP, Transformer, PPO
"""

import numpy as np
import pandas as pd

from ..models.baselines import (
    equal_weight, min_variance, mean_variance, hrp_allocation,
    risk_parity, max_diversification, ledoit_wolf_cov,
)
from ..training.trainer import dhrp_weights
from ..data.universe_config import get_universe_config

# Default backtest parameters
TRAIN_DAYS = 252
TEST_DAYS = 21
STEP_DAYS = 21
MIN_COVERAGE = 0.95
PURGE_DAYS = 5  # gap between train and test to prevent look-ahead
METHODS = [
    "EW", "MINVAR", "MV", "HRP", "RP", "MAXDIV",
    "DHRP", "LLM_DHRP", "MLP", "Transformer", "PPO",
]


def rolling_backtest(
    prices,
    is_em=False,
    dhrp_model=None,
    llm_dhrp_model=None,
    mlp_model=None,
    transformer_model=None,
    ppo_model=None,
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

    rets = prices.pct_change().dropna()
    results = []
    weights_history = []
    prev_weights = {}
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
        train = rets.iloc[train_end - train_days : train_end].fillna(0)
        if train.isna().mean().mean() > (1 - min_coverage):
            rebalance_idx += 1
            continue
        mu = train.mean().values * 252
        # Ledoit-Wolf shrinkage for better-conditioned covariance estimation
        try:
            cov = ledoit_wolf_cov(train)
        except Exception:
            cov = train.cov().values * 252
        cov = cov + np.eye(train.shape[1]) * cov_shrinkage
        test = rets.iloc[t : t + test_days].fillna(0)
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
                    text_features=text_features,
                    macro_features=macro_features,
                    rebalance_idx=rebalance_idx,
                    rebalance_date=rets.index[t],
                    volume=vol_window,
                )
                if w is None:
                    continue

                # Record weights for turnover analysis
                if return_weights:
                    weights_history.append({
                        "method": m,
                        "date": rets.index[t],
                        "weights": w.tolist(),
                    })

                # Transaction cost adjustment
                tc_drag = 0.0
                if transaction_cost_bps > 0 and m in prev_weights:
                    turnover = np.sum(np.abs(w - prev_weights[m]))
                    tc_drag = turnover * transaction_cost_bps / 10000
                prev_weights[m] = w.copy()

                for i, r in enumerate((test.values @ w).flatten()):
                    adj_r = float(r) - (tc_drag / test_days if i == 0 else 0)
                    results.append({"method": m, "date": test.index[i], "return": adj_r})
            except Exception:
                pass

        rebalance_idx += 1

    results_df = pd.DataFrame(results)
    if return_weights:
        return results_df, weights_history
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
    from .statistics import compute_stats

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
    text_features=None, macro_features=None,
    rebalance_idx=0, rebalance_date=None,
    volume=None,
):
    """Compute portfolio weights for a given method."""
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
        return dhrp_weights(dhrp_model, train_rets, is_em, volume=volume)
    elif method == "LLM_DHRP" and llm_dhrp_model is not None:
        return _llm_dhrp_weights(
            llm_dhrp_model, train_rets, is_em,
            text_features, macro_features,
            rebalance_idx, rebalance_date,
            volume=volume,
        )
    elif method == "MLP" and mlp_model is not None:
        return dhrp_weights(mlp_model, train_rets, is_em, volume=volume)
    elif method == "Transformer" and transformer_model is not None:
        return dhrp_weights(transformer_model, train_rets, is_em, volume=volume)
    elif method == "PPO" and ppo_model is not None:
        return dhrp_weights(ppo_model, train_rets, is_em, volume=volume)
    return None


def _llm_dhrp_weights(
    model, rets, is_em,
    text_features, macro_features,
    rebalance_idx, rebalance_date,
    volume=None,
):
    """Compute LLM-DHRP portfolio weights."""
    import torch
    from ..data.feature_engineering import make_features

    try:
        device = next(model.parameters()).device
        cov = (rets.cov().values * 252).astype(np.float32)
        if is_em:
            cov += np.eye(cov.shape[0]) * 0.01
        feat = make_features(rets, model.feature_dim, is_em, volume=volume).astype(np.float32)

        text_emb = None
        if text_features is not None and model.use_text:
            if "finbert" in text_features and text_features["finbert"] is not None:
                if rebalance_idx < len(text_features["finbert"]):
                    text_emb = text_features["finbert"][rebalance_idx].mean(axis=0)
                    text_emb = torch.from_numpy(text_emb.astype(np.float32)).to(device)

        macro_feat = None
        if macro_features is not None and model.use_macro and rebalance_date is not None:
            from ..data.fred_loader import get_macro_vector
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
        return w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
    except Exception as e:
        import warnings
        warnings.warn(f"_llm_dhrp_weights failed: {e}")
        return np.ones(rets.shape[1]) / rets.shape[1]
