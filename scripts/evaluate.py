#!/usr/bin/env python3
"""Evaluate a trained DHRP model: compute statistics, run statistical tests, generate figures."""

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.price_loader import load_universe, load_fama_french
from src.evaluation.statistics import (
    compute_stats, sharpe_difference_test, diebold_mariano_test,
)
from src.evaluation.factor_analysis import factor_analysis

warnings.filterwarnings("ignore")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate backtest results")
    parser.add_argument("--results", type=str, required=True, help="Path to results CSV")
    parser.add_argument("--config", type=str, default=None, help="Config YAML for metadata")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    results = pd.read_csv(args.results)
    if "date" in results.columns:
        results["date"] = pd.to_datetime(results["date"])

    # Determine if we have raw backtest data or summary stats
    if "return" in results.columns:
        print("=== RAW BACKTEST RESULTS ===")
        _evaluate_raw(results, args)
    else:
        print("=== SUMMARY STATS (from CSV) ===")
        print(results.to_string(index=False))


def _evaluate_raw(results, args):
    """Full evaluation on raw per-day returns."""
    methods = sorted(results["method"].unique())
    print(f"Methods: {methods}")
    print(f"Observations: {len(results)}")

    # 1. Performance stats
    print("\n--- Performance Statistics ---")
    stats = compute_stats(results)
    print(stats.to_string(index=False))

    # 2. Fama-French factor analysis (if config provides date range)
    if args.config:
        cfg = load_config(args.config)
        end = datetime.now()
        start = end - timedelta(days=cfg["start_offset_years"] * 365)
        try:
            ff = load_fama_french(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            factors = factor_analysis(results, ff)
            table = stats.merge(factors, on="Method", how="left").round(3)
            print("\n--- Combined Performance + Factor Analysis ---")
            print(table.to_string(index=False))
        except Exception as e:
            print(f"Warning: Factor analysis failed: {e}")
            table = stats

    # 3. Pairwise statistical tests
    print("\n--- Pairwise Statistical Tests ---")
    for i, m_a in enumerate(methods):
        for m_b in methods[i + 1 :]:
            try:
                diff = sharpe_difference_test(results, m_a, m_b)
                dm = diebold_mariano_test(results, m_a, m_b)
                sig = "*" if diff["bootstrap_p"] < 0.05 else ""
                print(
                    f"  {m_a} vs {m_b}: "
                    f"Sharpe diff={diff['sharpe_diff']:+.3f} "
                    f"(p={diff['bootstrap_p']:.3f}{sig}), "
                    f"DM={dm['DM_stat']:+.3f} (p={dm['p_value']:.3f})"
                )
            except Exception as e:
                print(f"  {m_a} vs {m_b}: test failed ({e})")

    # 4. Sub-period analysis
    print("\n--- Sub-period Analysis ---")
    results_ts = results.copy()
    results_ts["year"] = results_ts["date"].dt.year
    for year in sorted(results_ts["year"].unique()):
        yr_data = results_ts[results_ts["year"] == year]
        if len(yr_data) < 50:
            continue
        yr_stats = compute_stats(yr_data)
        best = yr_stats.loc[yr_stats["Sharpe"].idxmax()]
        print(f"  {year}: best={best['Method']} (Sharpe={best['Sharpe']:.3f})")

    # 5. Save full report
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{args.output_dir}/eval_report_{stamp}.csv"
    stats.to_csv(report_path, index=False)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
