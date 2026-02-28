#!/usr/bin/env python3
"""Run the full experiment: all 3 universes, all methods, all stats, all figures.

Usage:
    python scripts/run_full_experiment.py
    python scripts/run_full_experiment.py --universes DM EM
    python scripts/run_full_experiment.py --tc-bps 10
"""

import os
import sys
import warnings
from datetime import datetime, timedelta

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.price_loader import load_universe, load_fama_french, UNIVERSES
from src.training.trainer import train_dhrp
from src.evaluation.backtest import rolling_backtest
from src.evaluation.statistics import compute_stats, sharpe_difference_test, diebold_mariano_test
from src.evaluation.factor_analysis import factor_analysis


def run_universe(name, prices, ff, device="cpu", tc_bps=0):
    """Train + backtest + stats for one universe."""
    is_em = name in ("EM", "EM_small")
    n_assets = prices.shape[1]
    print(f"\n{'='*60}")
    print(f"  {name} UNIVERSE ({n_assets} assets)")
    print(f"{'='*60}")

    # Train DHRP
    print(f"\nTraining DHRP...")
    dhrp_model = train_dhrp(prices, device=device, is_em=is_em)

    # Backtest
    methods = ["EW", "MINVAR", "MV", "HRP", "RP", "MAXDIV", "DHRP"]
    print(f"\nBacktesting with {methods}...")
    results = rolling_backtest(
        prices, is_em=is_em, dhrp_model=dhrp_model, methods=methods,
        transaction_cost_bps=tc_bps,
    )
    print(f"  {len(results)} observations, {len(results['method'].unique())} methods")

    # Stats
    stats = compute_stats(results)
    try:
        factors = factor_analysis(results, ff)
        table = stats.merge(factors, on="Method", how="left").round(3)
    except Exception:
        table = stats.round(3)

    print(f"\n--- {name} Performance ---")
    print(table.to_string(index=False))

    # Statistical tests
    print(f"\n--- {name} Statistical Tests (vs HRP) ---")
    test_rows = []
    for m in sorted(results["method"].unique()):
        if m == "HRP":
            continue
        try:
            diff = sharpe_difference_test(results, m, "HRP")
            dm_test = diebold_mariano_test(results, m, "HRP")
            sig = "***" if diff["bootstrap_p"] < 0.01 else "**" if diff["bootstrap_p"] < 0.05 else "*" if diff["bootstrap_p"] < 0.10 else ""
            print(
                f"  {m:8s} vs HRP: diff={diff['sharpe_diff']:+.3f} "
                f"[{diff['bootstrap_ci_lo']:.3f}, {diff['bootstrap_ci_hi']:.3f}] "
                f"p={diff['bootstrap_p']:.3f} {sig} | "
                f"DM={dm_test['DM_stat']:+.3f} p={dm_test['p_value']:.3f}"
            )
            test_rows.append({
                "Method": m, "Sharpe_diff": diff["sharpe_diff"],
                "CI_lo": diff["bootstrap_ci_lo"], "CI_hi": diff["bootstrap_ci_hi"],
                "Boot_p": diff["bootstrap_p"], "DM_stat": dm_test["DM_stat"],
                "DM_p": dm_test["p_value"],
            })
        except Exception as e:
            print(f"  {m:8s} vs HRP: failed ({e})")

    return dhrp_model, results, table, pd.DataFrame(test_rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--universes", nargs="+", default=["DM", "EM", "Commodities"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--tc-bps", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)

    END = datetime.now().strftime("%Y-%m-%d")
    START = (datetime.now() - timedelta(days=10 * 365)).strftime("%Y-%m-%d")

    print(f"Period: {START} to {END}")
    print(f"Device: {device}")
    print(f"Transaction costs: {args.tc_bps} bps")
    print(f"Universes: {args.universes}")

    # Load FF factors once
    print("\nLoading Fama-French factors...")
    ff = load_fama_french(START, END)

    all_results = {}
    all_tables = {}
    all_prices = {}

    for univ in args.universes:
        print(f"\nLoading {univ} prices...")
        prices = load_universe(univ, START, END)
        if prices.empty or prices.shape[0] < 300:
            print(f"  Skipping {univ}: insufficient data ({prices.shape})")
            continue
        all_prices[univ] = prices

        model, results, table, tests = run_universe(
            univ, prices, ff, device=device, tc_bps=args.tc_bps,
        )
        all_results[univ] = results
        all_tables[univ] = table

        # Save
        torch.save(model.state_dict(), f"{args.output_dir}/models/dhrp_{univ}.pt")
        table.to_csv(f"{args.output_dir}/{univ}_results.csv", index=False)
        results.to_csv(f"{args.output_dir}/{univ}_returns.csv", index=False)
        if not tests.empty:
            tests.to_csv(f"{args.output_dir}/{univ}_tests.csv", index=False)

    # Generate figures if we have at least DM + EM
    if "DM" in all_results and "EM" in all_results:
        print("\n\nGenerating paper figures...")
        try:
            from src.visualization.plots import generate_all_figures
            generate_all_figures(
                all_results["DM"], all_results["EM"],
                all_tables["DM"], all_tables["EM"],
                all_prices["DM"], all_prices["EM"],
                output_dir=args.output_dir,
            )
        except Exception as e:
            print(f"  Figure generation failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE")
    print("=" * 60)
    for univ in args.universes:
        if univ in all_tables:
            best = all_tables[univ].loc[all_tables[univ]["Sharpe"].idxmax()]
            dhrp = all_tables[univ][all_tables[univ]["Method"] == "DHRP"]
            dhrp_sharpe = dhrp["Sharpe"].values[0] if len(dhrp) else "N/A"
            print(f"  {univ}: Best={best['Method']} (Sharpe={best['Sharpe']:.3f}), DHRP={dhrp_sharpe}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nAll results saved to {args.output_dir}/")
    print(f"Timestamp: {stamp}")


if __name__ == "__main__":
    main()
