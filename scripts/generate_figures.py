"""Generate all paper-ready figures from backtest results."""
import os
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load backtest results
results = {}
for u in ["DM", "EM", "Commodities"]:
    results[u] = pd.read_csv(f"results/full/{u}_backtest.csv")
    results[u]["date"] = pd.to_datetime(results[u]["date"])

out = "results/figures"
os.makedirs(out, exist_ok=True)

METHOD_ORDER = ["LLM_DHRP", "DHRP", "HRP", "EW", "MV", "MINVAR", "RP", "MAXDIV"]
METHOD_COLORS = {
    "LLM_DHRP": "#e74c3c", "DHRP": "#2980b9", "HRP": "#27ae60", "EW": "#8e44ad",
    "MV": "#f39c12", "MINVAR": "#1abc9c", "RP": "#e67e22", "MAXDIV": "#95a5a6",
}
METHOD_LABELS = {
    "LLM_DHRP": "LLM-DHRP (Ours)", "DHRP": "DHRP (Ours)", "HRP": "HRP",
    "EW": "Equal Weight", "MV": "Mean-Var", "MINVAR": "Min-Var",
    "RP": "Risk Parity", "MAXDIV": "Max Diversification",
}


def get_series(res, m):
    df = res[res["method"] == m].sort_values("date")
    return pd.Series(df["return"].values, index=df["date"].values)


# ============================================================
# FIGURE 1: Cumulative Returns (3 panels)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (uname, res) in zip(axes, results.items()):
    for m in METHOD_ORDER:
        if m in res["method"].values:
            s = get_series(res, m)
            cum = (1 + s).cumprod()
            lw = 2.5 if m in ["LLM_DHRP", "DHRP"] else 1.2
            alpha = 1.0 if m in ["LLM_DHRP", "DHRP", "HRP"] else 0.5
            ax.plot(cum.index, cum.values, label=METHOD_LABELS[m],
                    color=METHOD_COLORS[m], lw=lw, alpha=alpha)
    ax.set_title(uname, fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.set_ylabel("Cumulative Return (log)")
axes[0].legend(fontsize=7, loc="upper left")
plt.tight_layout()
plt.savefig(f"{out}/fig1_cumulative_returns.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 1: Cumulative returns")

# ============================================================
# FIGURE 2: Sharpe Ratio Comparison Bar Chart
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (uname, res) in zip(axes, results.items()):
    sharpes = []
    methods_present = []
    for m in METHOD_ORDER:
        if m in res["method"].values:
            r = res[res["method"] == m]["return"].values
            exc = r - 0.03 / 252
            sr = exc.mean() * 252 / (exc.std() * np.sqrt(252)) if exc.std() > 0 else 0
            sharpes.append(sr)
            methods_present.append(m)
    colors = [METHOD_COLORS[m] for m in methods_present]
    labels = [METHOD_LABELS[m] for m in methods_present]
    bars = ax.bar(range(len(sharpes)), sharpes, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(uname, fontsize=14, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(axis="y", alpha=0.3)
    best_idx = int(np.argmax(sharpes))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)
plt.tight_layout()
plt.savefig(f"{out}/fig2_sharpe_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 2: Sharpe comparison")

# ============================================================
# FIGURE 3: Rolling 1-Year Sharpe
# ============================================================
highlight = ["LLM_DHRP", "DHRP", "HRP", "EW"]
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
for ax, (uname, res) in zip(axes, results.items()):
    for m in highlight:
        if m in res["method"].values:
            s = get_series(res, m)
            roll = s.rolling(252)
            rs = roll.mean() / roll.std() * np.sqrt(252)
            lw = 2.5 if m in ["LLM_DHRP", "DHRP"] else 1.2
            ax.plot(rs.index, rs.values, label=METHOD_LABELS[m],
                    color=METHOD_COLORS[m], lw=lw)
    ax.axhline(0, color="black", ls="--", alpha=0.5)
    ax.set_title(f"{uname}: Rolling 1-Year Sharpe", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    ax.set_ylabel("Sharpe Ratio")
plt.tight_layout()
plt.savefig(f"{out}/fig3_rolling_sharpe.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 3: Rolling Sharpe")

# ============================================================
# FIGURE 4: Drawdowns
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
for ax, (uname, res) in zip(axes, results.items()):
    for m in highlight:
        if m in res["method"].values:
            s = get_series(res, m)
            cum = (1 + s).cumprod()
            dd = (cum - cum.expanding().max()) / cum.expanding().max() * 100
            lw = 2.0 if m in ["LLM_DHRP", "DHRP"] else 1.0
            ax.fill_between(dd.index, dd.values, 0, alpha=0.15, color=METHOD_COLORS[m])
            ax.plot(dd.index, dd.values, label=METHOD_LABELS[m],
                    color=METHOD_COLORS[m], lw=lw)
    ax.set_title(f"{uname}: Drawdown", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_ylabel("Drawdown (%)")
plt.tight_layout()
plt.savefig(f"{out}/fig4_drawdowns.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 4: Drawdowns")

# ============================================================
# FIGURE 5: Risk-Return Scatter
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (uname, res) in zip(axes, results.items()):
    for m in METHOD_ORDER:
        if m in res["method"].values:
            r = res[res["method"] == m]["return"].values
            ret = r.mean() * 252 * 100
            vol = r.std() * np.sqrt(252) * 100
            s = 250 if m in ["LLM_DHRP", "DHRP"] else 120
            ax.scatter(vol, ret, s=s, color=METHOD_COLORS[m], edgecolors="black",
                       lw=1.5, zorder=5, alpha=0.85)
            label = METHOD_LABELS[m].replace(" (Ours)", "")
            ax.annotate(label, (vol, ret), fontsize=8, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title(uname, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out}/fig5_risk_return.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 5: Risk-return scatter")

# ============================================================
# FIGURE 6: LLM-DHRP vs DHRP Delta Analysis
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (uname, res) in zip(axes, results.items()):
    s_llm = get_series(res, "LLM_DHRP")
    s_dhrp = get_series(res, "DHRP")
    common = s_llm.index.intersection(s_dhrp.index)
    diff = s_llm.loc[common] - s_dhrp.loc[common]
    cum_diff = diff.cumsum() * 100
    ax.fill_between(common, cum_diff.values, 0,
                    where=cum_diff.values >= 0, alpha=0.3, color="green")
    ax.fill_between(common, cum_diff.values, 0,
                    where=cum_diff.values < 0, alpha=0.3, color="red")
    ax.plot(common, cum_diff.values, color="black", lw=1.5)
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_title(f"{uname}: LLM-DHRP minus DHRP", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Excess Return (%)")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out}/fig6_llm_delta.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 6: LLM delta analysis")

# ============================================================
# FIGURE 7: Architecture Diagram (text-based)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# Boxes
boxes = [
    (0.5, 4.5, "Financial\nHeadlines", "#ffeaa7"),
    (0.5, 2.5, "Price\nFeatures (48d)", "#74b9ff"),
    (0.5, 0.5, "FRED\nMacro (8d)", "#a29bfe"),
    (3, 4.5, "FinBERT\n(768d)", "#fd79a8"),
    (3, 2.5, "Text\nProjection", "#fdcb6e"),
    (5.5, 3.5, "Cross-Modal\nFusion", "#55efc4"),
    (5.5, 1.5, "Macro\nGating", "#dfe6e9"),
    (8, 3, "DHRP\nSoft-Gating\nTree", "#e17055"),
    (8, 0.5, "Portfolio\nWeights", "#00b894"),
]
for x, y, text, color in boxes:
    ax.add_patch(plt.Rectangle((x-0.6, y-0.4), 1.2, 0.8,
                               facecolor=color, edgecolor="black", lw=1.5, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)

# Arrows
arrows = [
    (1.1, 4.5, 2.4, 4.5), (1.1, 2.5, 2.4, 2.5), (1.1, 0.5, 4.9, 1.5),
    (3.6, 4.5, 4.9, 3.7), (3.6, 2.5, 4.9, 3.3),
    (6.1, 3.5, 7.4, 3.2), (6.1, 1.5, 7.4, 2.8),
    (8, 2.6, 8, 0.9),
]
for x1, y1, x2, y2 in arrows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

ax.set_title("LLM-DHRP Architecture", fontsize=16, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(f"{out}/fig7_architecture.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 7: Architecture diagram")

print(f"\nAll 7 figures saved to {out}/")
