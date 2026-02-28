"""Plotting utilities for paper figures."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS = {
    "LLM_DHRP": "#e74c3c", "DHRP": "#2980b9", "HRP": "#27ae60", "EW": "#8e44ad",
    "MV": "#f39c12", "MINVAR": "#1abc9c", "RP": "#e67e22", "MAXDIV": "#95a5a6",
}

METHOD_LABELS = {
    "LLM_DHRP": "LLM-DHRP", "DHRP": "DHRP", "HRP": "HRP",
    "EW": "Equal Weight", "MV": "Mean-Var", "MINVAR": "Min-Var",
    "RP": "Risk Parity", "MAXDIV": "Max Diversification",
}


def get_series(res, m):
    """Extract a return series for a single method."""
    df = res[res["method"] == m].sort_values("date")
    return pd.Series(df["return"].values, index=pd.to_datetime(df["date"].values))


def plot_cumulative(results_dict, output_dir="results/figures"):
    """Plot cumulative returns for multiple universes."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(results_dict), figsize=(6 * len(results_dict), 5))
    if len(results_dict) == 1:
        axes = [axes]
    for ax, (uname, res) in zip(axes, results_dict.items()):
        for m in sorted(res["method"].unique()):
            s = get_series(res, m)
            cum = (1 + s).cumprod()
            lw = 2.5 if m in ["LLM_DHRP", "DHRP"] else 1.0
            color = METHOD_COLORS.get(m, "gray")
            ax.plot(cum.index, cum.values, label=METHOD_LABELS.get(m, m),
                    color=color, lw=lw, alpha=0.8 if lw > 1 else 0.5)
        ax.set_title(uname, fontsize=14, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.set_ylabel("Cumulative Return (log)")
    axes[0].legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cumulative_returns.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_sharpe_bars(results_dict, output_dir="results/figures"):
    """Plot Sharpe ratio bar charts for multiple universes."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(results_dict), figsize=(6 * len(results_dict), 5))
    if len(results_dict) == 1:
        axes = [axes]
    for ax, (uname, res) in zip(axes, results_dict.items()):
        methods_present = sorted(res["method"].unique())
        sharpes = []
        for m in methods_present:
            r = res[res["method"] == m]["return"].values
            exc = r - 0.03 / 252
            sr = exc.mean() * 252 / (exc.std() * np.sqrt(252)) if exc.std() > 0 else 0
            sharpes.append(sr)
        colors = [METHOD_COLORS.get(m, "gray") for m in methods_present]
        labels = [METHOD_LABELS.get(m, m) for m in methods_present]
        ax.bar(range(len(sharpes)), sharpes, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title(uname, fontsize=14, fontweight="bold")
        ax.axhline(0, color="black", lw=0.8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sharpe_bars.png", dpi=300, bbox_inches="tight")
    plt.close()
