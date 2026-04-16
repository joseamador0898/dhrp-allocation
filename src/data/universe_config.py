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
        "tree_depth": 3,
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
        "tree_depth": 3,
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
    # Tree depth 3 captures the cyclical/defensive/growth structure.
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
        "tree_depth": 3,
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
