#!/usr/bin/env python3
"""Train DHRP / LLM-DHRP and run rolling backtest with all methods."""

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.price_loader import load_universe, load_fama_french
from src.training.trainer import train_dhrp, train_llm_dhrp
from src.evaluation.backtest import rolling_backtest
from src.evaluation.statistics import compute_stats, sharpe_difference_test, diebold_mariano_test
from src.evaluation.factor_analysis import factor_analysis

warnings.filterwarnings("ignore")


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train DHRP/LLM-DHRP and run backtest")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--text-features", type=str, default=None,
                        help="Path to precomputed text features (.npz)")
    parser.add_argument("--macro", action="store_true", help="Use FRED macro features")
    parser.add_argument("--tc-bps", type=int, default=0,
                        help="Transaction costs in basis points")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine date range
    end = datetime.now()
    start = end - timedelta(days=cfg["start_offset_years"] * 365)
    start_str, end_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    is_em = cfg["is_em"]
    universe_name = cfg["universe"]

    print(f"=== Configuration ===")
    print(f"Universe: {universe_name}, EM: {is_em}")
    print(f"Period: {start_str} to {end_str}")
    print(f"Device: {device}")
    if args.tc_bps > 0:
        print(f"Transaction costs: {args.tc_bps} bps")
    print()

    # Load data
    print("Loading data...")
    prices = load_universe(universe_name, start_str, end_str)
    ff = load_fama_french(start_str, end_str)

    # Load text features if provided
    text_features = None
    if args.text_features and os.path.exists(args.text_features):
        data = np.load(args.text_features)
        text_features = {k: data[k] for k in data.files}
        print(f"Loaded text features from {args.text_features}")
        for k, v in text_features.items():
            print(f"  {k}: {v.shape}")

    # Load macro features if requested
    macro_df = None
    if args.macro:
        try:
            from src.data.fred_loader import load_fred_data, make_macro_features
            print("Loading FRED macro features...")
            fred_df = load_fred_data(start_str, end_str)
            if not fred_df.empty:
                macro_df = make_macro_features(fred_df)
                print(f"  Macro features: {macro_df.shape}")
            else:
                print("  Warning: FRED data empty (check FRED_API_KEY)")
        except Exception as e:
            print(f"  Warning: FRED loading failed: {e}")

    # Train DHRP
    print(f"\nTraining DHRP ({device})...")
    dhrp_model = train_dhrp(prices, device=device, is_em=is_em)

    # Train LLM-DHRP if text features are available
    llm_dhrp_model = None
    methods = cfg.get("methods", ["EW", "MINVAR", "MV", "HRP", "DHRP"])
    llm_cfg = cfg.get("llm_dhrp", {})

    if text_features is not None or llm_cfg.get("enabled", False):
        print(f"\nTraining LLM-DHRP ({device})...")
        llm_dhrp_model = train_llm_dhrp(
            prices,
            text_features=text_features,
            macro_features=macro_df.values if macro_df is not None else None,
            device=device,
            is_em=is_em,
            text_dim=llm_cfg.get("text_dim", 768),
            use_text=text_features is not None,
            use_macro=macro_df is not None,
            macro_dim=macro_df.shape[1] if macro_df is not None else 4,
            fusion_type=llm_cfg.get("fusion_type", "cross_attention"),
            depth=llm_cfg.get("depth", 3),
            hidden_dim=llm_cfg.get("hidden_dim", 64),
            epochs=llm_cfg.get("epochs", 60),
            lr=llm_cfg.get("lr", 3e-4),
        )
        if "LLM_DHRP" not in methods:
            methods.append("LLM_DHRP")

    # Run backtest
    bt_cfg = cfg.get("backtest", {})
    print(f"\nBacktesting with methods: {methods}...")
    results = rolling_backtest(
        prices, is_em=is_em,
        dhrp_model=dhrp_model,
        llm_dhrp_model=llm_dhrp_model,
        text_features=text_features,
        macro_features=macro_df,
        methods=methods,
        train_days=bt_cfg.get("train_days", 252),
        test_days=bt_cfg.get("test_days", 21),
        step_days=bt_cfg.get("step_days", 21),
        transaction_cost_bps=args.tc_bps,
    )
    print(f"  {len(results)} observations")

    # Compute statistics
    print(f"\n=== {universe_name.upper()} RESULTS ===")
    stats = compute_stats(results)
    factors = factor_analysis(results, ff)
    table = stats.merge(factors, on="Method", how="left").round(3)
    print(table.to_string(index=False))

    # Statistical tests — compare each model vs HRP
    print("\n=== STATISTICAL TESTS ===")
    test_methods = [m for m in methods if m not in ("EW", "HRP")]
    for m in test_methods:
        if m in results["method"].unique() and "HRP" in results["method"].unique():
            try:
                diff = sharpe_difference_test(results, m, "HRP")
                dm_test = diebold_mariano_test(results, m, "HRP")
                sig = " *" if diff["bootstrap_p"] < 0.05 else ""
                print(f"{m} vs HRP: diff={diff['sharpe_diff']:+.3f}, "
                      f"CI=[{diff['bootstrap_ci_lo']:.3f}, {diff['bootstrap_ci_hi']:.3f}], "
                      f"p={diff['bootstrap_p']:.4f}{sig} | "
                      f"DM={dm_test['DM_stat']:+.3f} (p={dm_test['p_value']:.4f})")
            except Exception as e:
                print(f"{m} vs HRP: test failed ({e})")

    # Export
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table.to_csv(f"{args.output_dir}/{universe_name}_{stamp}.csv", index=False)
    print(f"\nSaved to {args.output_dir}/{universe_name}_{stamp}.csv")

    # Save models
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(dhrp_model.state_dict(), f"{model_dir}/dhrp_{universe_name}.pt")
    if llm_dhrp_model is not None:
        torch.save(llm_dhrp_model.state_dict(), f"{model_dir}/llm_dhrp_{universe_name}.pt")
    print(f"Models saved to {model_dir}/")


if __name__ == "__main__":
    main()
