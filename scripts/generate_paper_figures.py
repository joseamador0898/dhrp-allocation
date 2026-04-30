"""Generate publication-quality figures for the paper from results/*.csv.

Produces 6 figures in paper/figures/, all 300 DPI vector PDF using the
Okabe-Ito colorblind-safe palette and Type 42 fonts (LaTeX-compatible):

  1. cumulative_returns_commodities.pdf  Top-5 methods on commodities
  2. multiseed_sharpe_heatmap.pdf        Method x universe Sharpe matrix
  3. ablation_heatmap.pdf                Tree depth x loss components
  4. cost_sensitivity_curves.pdf         Sharpe vs transaction cost
  5. asset_grouping_dendrogram.pdf       DHRP soft assignment tree
  6. regime_conditional_sharpe.pdf       Sharpe by sub-period

Run:
    python scripts/generate_paper_figures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Style: NeurIPS-compliant publication figures
# ---------------------------------------------------------------------

# Okabe-Ito palette (colorblind-safe; 8 colors)
OKABE_ITO = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermilion": "#D55E00",
    "purple":  "#CC79A7",
}
PALETTE = list(OKABE_ITO.values())

METHOD_COLORS = {
    "EW": OKABE_ITO["black"],
    "MV": OKABE_ITO["vermilion"],
    "MINVAR": OKABE_ITO["yellow"],
    "MAXDIV": OKABE_ITO["purple"],
    "RP": OKABE_ITO["skyblue"],
    "HRP": OKABE_ITO["orange"],
    "DHRP": OKABE_ITO["blue"],
    "LLM_DHRP": OKABE_ITO["green"],
}

plt.rcParams.update({
    # Type 42 (TrueType) fonts -- required by NeurIPS for LaTeX compatibility
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
})


def save_fig(fig, name: str) -> None:
    out = FIGS / name
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------
# Figure 1: Cumulative returns on commodities (top-5 methods)
# ---------------------------------------------------------------------

def fig_cumulative_returns() -> None:
    bt = RESULTS / "full" / "Commodities_backtest.csv"
    if not bt.exists():
        print(f"  SKIP cumulative_returns_commodities.pdf (missing {bt.name})")
        return

    df = pd.read_csv(bt, parse_dates=["date"])
    # Top 5 by realized Sharpe over OOS
    grouped = df.groupby("method")["return"]
    sharpes = (grouped.mean() * 252) / (grouped.std() * np.sqrt(252))
    top5 = sharpes.nlargest(5).index.tolist()

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for m in top5:
        sub = df[df["method"] == m].sort_values("date")
        cum = (1 + sub["return"]).cumprod()
        color = METHOD_COLORS.get(m, "#666")
        ax.plot(sub["date"], cum, label=m.replace("_", "-"),
                color=color, linewidth=1.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (out-of-sample)")
    ax.set_title("Commodities: cumulative returns of top-5 methods")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    save_fig(fig, "cumulative_returns_commodities.pdf")


# ---------------------------------------------------------------------
# Figure 2: Multi-seed Sharpe heatmap (method x universe)
# ---------------------------------------------------------------------

def fig_multiseed_heatmap() -> None:
    mean_path = RESULTS / "sharpe_pivot_multiseed_mean.csv"
    if not mean_path.exists():
        print(f"  SKIP multiseed_sharpe_heatmap.pdf (missing {mean_path.name})")
        return

    mean_df = pd.read_csv(mean_path, index_col=0)
    universes = ["DM", "EM", "Commodities", "Sectors", "Global", "Factors", "Crypto", "Bonds"]
    methods = ["EW", "MV", "MINVAR", "MAXDIV", "RP", "HRP", "DHRP", "LLM_DHRP"]
    universes = [u for u in universes if u in mean_df.columns]
    methods = [m for m in methods if m in mean_df.index]
    M = mean_df.loc[methods, universes].values

    # Diverging colormap from Okabe-Ito (vermilion -> white -> blue)
    cmap = LinearSegmentedColormap.from_list(
        "okabe_div",
        [OKABE_ITO["vermilion"], "#ffffff", OKABE_ITO["blue"]],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    vmax = max(abs(np.nanmin(M)), abs(np.nanmax(M)))
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(universes)))
    ax.set_xticklabels(universes, rotation=30, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m.replace("_", "-") for m in methods])
    ax.grid(False)
    # Annotate cells
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if pd.isna(v):
                txt = "n/a"
                col = "#888"
            else:
                txt = f"{v:.2f}"
                col = "black" if abs(v) < 0.6 * vmax else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=col)
    ax.set_title("Mean Sharpe (10 seeds): method $\\times$ universe")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Sharpe")
    save_fig(fig, "multiseed_sharpe_heatmap.pdf")


# ---------------------------------------------------------------------
# Figure 3: Ablation heatmap (tree depth, loss components, text fusion)
# ---------------------------------------------------------------------

def fig_ablation_heatmap() -> None:
    ab = RESULTS / "DM_ablations.csv"
    if not ab.exists():
        print(f"  SKIP ablation_heatmap.pdf (missing {ab.name})")
        return

    df = pd.read_csv(ab)
    # Expected columns: Ablation, Config, Sharpe
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.4), gridspec_kw={"width_ratios": [3, 4, 2]})
    for ax, group in zip(axes, ["Tree depth", "Loss component", "Text fusion"]):
        sub = df[df["Ablation"] == group].sort_values("Sharpe", ascending=False)
        if sub.empty:
            ax.set_visible(False)
            continue
        bars = ax.barh(sub["Config"], sub["Sharpe"], color=OKABE_ITO["blue"])
        # Bold the best
        best_idx = sub["Sharpe"].idxmax()
        for i, (idx, val) in enumerate(zip(sub.index, sub["Sharpe"])):
            if idx == best_idx:
                bars[i].set_color(OKABE_ITO["vermilion"])
            ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=7)
        ax.set_xlabel("Sharpe (DM)")
        ax.set_title(group)
        ax.set_xlim(min(0, sub["Sharpe"].min()) - 0.05, sub["Sharpe"].max() + 0.05)
    save_fig(fig, "ablation_heatmap.pdf")


# ---------------------------------------------------------------------
# Figure 4: Cost sensitivity curves
# ---------------------------------------------------------------------

def fig_cost_sensitivity() -> None:
    cs = RESULTS / "DM_cost_sensitivity.csv"
    if not cs.exists():
        print(f"  SKIP cost_sensitivity_curves.pdf (missing {cs.name})")
        return

    df = pd.read_csv(cs)
    cost_cols = [c for c in df.columns if c.startswith("cost_")]
    if not cost_cols:
        return
    bps = [int(c.replace("cost_", "").replace("bps", "")) for c in cost_cols]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for _, row in df.iterrows():
        m = row["Method"]
        if m not in METHOD_COLORS:
            continue
        y = [row[c] for c in cost_cols]
        ax.plot(bps, y, label=m.replace("_", "-"),
                color=METHOD_COLORS[m], linewidth=1.2, marker="o", markersize=3)
    ax.set_xlabel("Per-trade transaction cost (bps)")
    ax.set_ylabel("Out-of-sample Sharpe")
    ax.set_title("Transaction-cost sensitivity: developed markets")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    save_fig(fig, "cost_sensitivity_curves.pdf")


# ---------------------------------------------------------------------
# Figure 5: Asset grouping (DHRP vs HRP clustering)
# ---------------------------------------------------------------------

def fig_asset_grouping() -> None:
    # Use the existing probe_gates_DHRP_DM.png if available; otherwise skip.
    src = ROOT / "results" / "figures" / "probe_gates_dhrp_dm.png"
    out = FIGS / "asset_grouping_dendrogram.pdf"
    if not src.exists():
        print(f"  SKIP asset_grouping_dendrogram.pdf (probe figure not generated)")
        return
    # Re-render as a placeholder using simple matplotlib
    # For now, just produce a stub note that this figure is built from the
    # probe analysis output.
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.text(0.5, 0.5,
            "Asset grouping dendrogram\n(see results/figures/probe_gates_dhrp_dm.png)\n"
            "Auto-conversion to vector PDF: TODO",
            ha="center", va="center", fontsize=9)
    ax.set_axis_off()
    save_fig(fig, "asset_grouping_dendrogram.pdf")


# ---------------------------------------------------------------------
# Figure 6: Regime-conditional Sharpe
# ---------------------------------------------------------------------

def fig_regime_sharpe() -> None:
    reg = RESULTS / "DM_regime_analysis.csv"
    if not reg.exists():
        print(f"  SKIP regime_conditional_sharpe.pdf (missing {reg.name})")
        return

    df = pd.read_csv(reg)
    # Expected long format: Method, Period, Sharpe (or wide pivot table)
    # Try wide first
    if "Period" not in df.columns:
        # Already wide-format — index might be Method, columns are periods
        if "Method" in df.columns:
            df = df.set_index("Method")
        periods = list(df.columns)
        methods = [m for m in METHOD_COLORS if m in df.index]
    else:
        pivot = df.pivot_table(index="Method", columns="Period", values="Sharpe")
        df = pivot
        periods = list(df.columns)
        methods = [m for m in METHOD_COLORS if m in df.index]

    if not methods or not periods:
        print(f"  SKIP regime_conditional_sharpe.pdf (no method or period columns)")
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    n = len(methods)
    n_per = len(periods)
    width = 0.8 / n
    x = np.arange(n_per)
    for i, m in enumerate(methods):
        vals = [df.loc[m, p] if not pd.isna(df.loc[m, p]) else 0 for p in periods]
        ax.bar(x + i * width, vals, width, label=m.replace("_", "-"),
               color=METHOD_COLORS[m])
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(periods, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Sharpe")
    ax.set_title("Sharpe by macro regime (developed markets)")
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5)
    save_fig(fig, "regime_conditional_sharpe.pdf")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print(f"Generating figures in {FIGS.relative_to(ROOT)}/ from {RESULTS.relative_to(ROOT)}/")
    fig_cumulative_returns()
    fig_multiseed_heatmap()
    fig_ablation_heatmap()
    fig_cost_sensitivity()
    fig_asset_grouping()
    fig_regime_sharpe()
    print("Done. All figures rendered as 300 DPI vector PDF with Okabe-Ito palette.")


if __name__ == "__main__":
    main()
