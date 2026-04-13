"""Plotting utilities for paper figures (NeurIPS-compatible)."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# NeurIPS-compatible defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
})

COLUMN_WIDTH = 5.5  # NeurIPS single-column width in inches

METHOD_COLORS = {
    "LLM_DHRP": "#e74c3c", "DHRP": "#2980b9", "HRP": "#27ae60", "EW": "#8e44ad",
    "MV": "#f39c12", "MINVAR": "#1abc9c", "RP": "#e67e22", "MAXDIV": "#95a5a6",
    "MLP": "#d35400", "Transformer": "#2c3e50", "PPO": "#c0392b",
}

METHOD_LABELS = {
    "LLM_DHRP": "LLM-DHRP", "DHRP": "DHRP", "HRP": "HRP",
    "EW": "Equal Weight", "MV": "Mean-Var", "MINVAR": "Min-Var",
    "RP": "Risk Parity", "MAXDIV": "Max Diversification",
    "MLP": "MLP", "Transformer": "Transformer", "PPO": "PPO",
}


def get_series(res, m):
    """Extract a return series for a single method.

    Defensively averages duplicate (method, date) rows so cumulative-return
    plots never compound the same day twice. With the post-fix
    rolling_backtest this is a no-op.
    """
    df = res[res["method"] == m]
    if df.empty:
        return pd.Series(dtype=float)
    df = df.groupby("date", as_index=True)["return"].mean().sort_index()
    return pd.Series(df.values, index=pd.to_datetime(df.index))


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
            # Dedup duplicate dates (see get_series rationale)
            r = get_series(res, m).values
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


# ---------------------------------------------------------------------------
# New publication-grade figures
# ---------------------------------------------------------------------------

def plot_ablation_heatmap(ablation_df, metric="Sharpe", output_dir="results/figures"):
    """Plot heatmap of ablation results.

    Args:
        ablation_df: DataFrame with columns [Ablation, Config, Sharpe, Sortino, ...]
        metric: which metric column to plot
        output_dir: output directory
    """
    try:
        import seaborn as sns
    except ImportError:
        print("seaborn required for heatmap plots")
        return

    os.makedirs(output_dir, exist_ok=True)
    pivot = ablation_df.pivot(index="Ablation", columns="Config", values=metric)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=pivot.values.mean(),
                linewidths=0.5, ax=ax, cbar_kws={"label": metric, "shrink": 0.8})
    ax.set_title(f"Ablation Study: {metric}", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_regime_bars(regime_df, metric="Sharpe", output_dir="results/figures"):
    """Plot grouped bar chart of per-regime performance.

    Args:
        regime_df: DataFrame from subperiod_analysis() with columns
                   [Method, Period, Sharpe, Sortino, MaxDD, CVaR_5, N]
        metric: which metric to plot
        output_dir: output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    if regime_df.empty:
        return

    methods = sorted(regime_df["Method"].unique())
    periods = regime_df["Period"].unique()
    n_methods = len(methods)
    n_periods = len(periods)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 1.5, 4))
    x = np.arange(n_periods)
    width = 0.8 / n_methods

    for i, m in enumerate(methods):
        mdf = regime_df[regime_df["Method"] == m]
        vals = []
        for p in periods:
            row = mdf[mdf["Period"] == p]
            vals.append(row[metric].values[0] if not row.empty else 0)
        color = METHOD_COLORS.get(m, "gray")
        ax.bar(x + i * width, vals, width, label=METHOD_LABELS.get(m, m),
               color=color, edgecolor="black", linewidth=0.3)

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(periods, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(metric)
    ax.set_title(f"Regime-Conditional {metric}", fontweight="bold")
    ax.legend(fontsize=6, ncol=min(n_methods, 4), loc="upper right")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/regime_bars.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_cost_sensitivity(cost_df, output_dir="results/figures"):
    """Plot Sharpe ratio vs transaction cost (bps) for each method.

    Args:
        cost_df: DataFrame from cost_sensitivity_analysis() with columns
                 [Method, cost_0bps, cost_5bps, cost_10bps, ...]
        output_dir: output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    if cost_df.empty:
        return

    cost_cols = [c for c in cost_df.columns if c.startswith("cost_")]
    bps_vals = [int(c.replace("cost_", "").replace("bps", "")) for c in cost_cols]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.5))
    for _, row in cost_df.iterrows():
        m = row["Method"]
        vals = [row[c] for c in cost_cols]
        color = METHOD_COLORS.get(m, "gray")
        lw = 2.0 if m in ["DHRP", "LLM_DHRP"] else 1.0
        ax.plot(bps_vals, vals, marker="o", markersize=3, label=METHOD_LABELS.get(m, m),
                color=color, lw=lw)

    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Cost Sensitivity Analysis", fontweight="bold")
    ax.legend(fontsize=6, ncol=2, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cost_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_seed_boxplots(seed_results, metric="Sharpe", output_dir="results/figures"):
    """Box-and-whisker plots of a metric across random seeds.

    Args:
        seed_results: list of DataFrames from compute_stats() (one per seed)
        metric: which metric to plot
        output_dir: output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    if not seed_results:
        return

    # Collect metric per method across seeds
    data = {}
    for sdf in seed_results:
        for _, row in sdf.iterrows():
            m = row["Method"]
            if m not in data:
                data[m] = []
            val = row.get(metric, np.nan)
            if not np.isnan(val):
                data[m].append(val)

    methods = sorted(data.keys())
    box_data = [data[m] for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    colors = [METHOD_COLORS.get(m, "gray") for m in methods]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.5))
    bp = ax.boxplot(box_data, patch_artist=True, labels=labels,
                    widths=0.5, medianprops={"color": "black", "lw": 1.5})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_ylabel(metric)
    ax.set_title(f"Multi-Seed Robustness: {metric}", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/seed_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_pairwise_dm_heatmap(dm_df, output_dir="results/figures"):
    """Heatmap of pairwise Diebold-Mariano test p-values.

    Args:
        dm_df: DataFrame with columns [method_a, method_b, p_value, adjusted_p]
        output_dir: output directory
    """
    try:
        import seaborn as sns
    except ImportError:
        print("seaborn required for heatmap plots")
        return

    os.makedirs(output_dir, exist_ok=True)
    if dm_df.empty:
        return

    # Build symmetric p-value matrix
    all_methods = sorted(set(dm_df["method_a"].unique()) | set(dm_df["method_b"].unique()))
    n = len(all_methods)
    pmat = np.ones((n, n))
    for _, row in dm_df.iterrows():
        i = all_methods.index(row["method_a"])
        j = all_methods.index(row["method_b"])
        pmat[i, j] = row.get("adjusted_p", row["p_value"])
        pmat[j, i] = pmat[i, j]

    labels = [METHOD_LABELS.get(m, m) for m in all_methods]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.8))
    sns.heatmap(pmat, annot=True, fmt=".3f", xticklabels=labels, yticklabels=labels,
                cmap="RdYlGn_r", vmin=0, vmax=0.1, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Adjusted p-value", "shrink": 0.8})
    ax.set_title("Pairwise DM Test (Holm-Bonferroni)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pairwise_dm_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
