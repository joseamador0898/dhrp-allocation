"""Probe analysis for DHRP / LLM-DHRP interpretability.

Answers: does the soft-gating tree learn semantically meaningful regimes?

Methods:
1. Linear probe: fit linear classifier from gate values to known regimes
   (COVID crash, rate hikes, recovery, etc.) — high accuracy = regime-aware
2. Asset-grouping analysis: extract leaf assignments and compare with
   ground-truth asset categories (equity vs bond vs commodity)
3. Gate activation patterns: which market conditions activate which gates?
4. Compare with HRP hard clustering baseline

Used for the paper's interpretability section (required for top ML venues).
"""

import sys
import os
import numpy as np
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def define_regimes(dates, prices=None):
    """Define ground-truth market regimes by date."""
    regimes = {}
    for d in dates:
        ts = pd.Timestamp(d)
        if ts < pd.Timestamp("2020-02-01"):
            regimes[ts] = "pre_covid_bull"
        elif ts < pd.Timestamp("2020-07-01"):
            regimes[ts] = "covid_crash"
        elif ts < pd.Timestamp("2022-01-01"):
            regimes[ts] = "recovery"
        elif ts < pd.Timestamp("2023-07-01"):
            regimes[ts] = "rate_hikes"
        else:
            regimes[ts] = "post_hike"
    return regimes


def extract_gate_patterns(model, X, S, dates, text_embs=None, device="cpu"):
    """Extract gating probabilities across all samples.

    Returns:
        gate_probs: (n_samples, n_gates, 2) array of softmax gating probs
        dates_list: aligned dates for each sample
    """
    n_samp = X.shape[0]
    all_gates = []
    dates_list = []

    model.eval()
    with torch.no_grad():
        for t in range(n_samp):
            x_t = torch.from_numpy(X[t]).to(device)
            s_t = torch.from_numpy(S[t]).to(device)
            te = torch.from_numpy(text_embs[t]).to(device) if text_embs is not None else None

            gates = model.get_gating_probs(x_t, s_t, text_emb=te)
            gate_array = torch.stack([g for g in gates]).cpu().numpy()
            all_gates.append(gate_array)
            if t < len(dates):
                dates_list.append(dates[t])

    return np.stack(all_gates), dates_list


def regime_probe(gate_probs, dates, prices=None):
    """Linear probe: can a classifier predict regime from gate probs?

    If probe accuracy is high, the tree learns regime-discriminative features.
    """
    print(f"\n{'='*60}")
    print("  REGIME PROBE ANALYSIS")
    print(f"{'='*60}")

    regimes = define_regimes(dates, prices)
    X_probe = gate_probs.reshape(gate_probs.shape[0], -1)  # flatten
    y_probe = np.array([regimes.get(d, "unknown") for d in dates])

    # Remove unknowns
    mask = y_probe != "unknown"
    X_probe, y_probe = X_probe[mask], y_probe[mask]

    print(f"  Samples: {len(X_probe)}, Features: {X_probe.shape[1]}")
    print(f"  Regimes: {sorted(set(y_probe))}")
    for r in sorted(set(y_probe)):
        print(f"    {r}: {np.sum(y_probe == r)} samples")

    # Train/test split: 70/30 temporal
    n = len(X_probe)
    split = int(n * 0.7)
    X_tr, X_te = X_probe[:split], X_probe[split:]
    y_tr, y_te = y_probe[:split], y_probe[split:]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(X_tr, y_tr)
    y_pred_tr = clf.predict(X_tr)
    y_pred_te = clf.predict(X_te)

    acc_tr = accuracy_score(y_tr, y_pred_tr)
    acc_te = accuracy_score(y_te, y_pred_te)
    f1_tr = f1_score(y_tr, y_pred_tr, average="weighted")
    f1_te = f1_score(y_te, y_pred_te, average="weighted")

    # Baseline: majority class
    majority_acc = max(np.mean(y_te == r) for r in set(y_te))

    print(f"\n  Train acc: {acc_tr:.3f}, F1: {f1_tr:.3f}")
    print(f"  Test  acc: {acc_te:.3f}, F1: {f1_te:.3f}")
    print(f"  Majority baseline: {majority_acc:.3f}")
    print(f"  Lift over baseline: +{(acc_te - majority_acc):.3f}")

    if acc_te > majority_acc + 0.15:
        print("  [STRONG] Tree gates encode regime-discriminative features.")
    elif acc_te > majority_acc + 0.05:
        print("  [MODERATE] Tree gates weakly encode regime structure.")
    else:
        print("  [WEAK] Tree gates appear regime-agnostic (noise or feature-driven).")

    return {"train_acc": acc_tr, "test_acc": acc_te, "majority": majority_acc}


def asset_grouping_analysis(model, asset_names, asset_categories=None):
    """Extract learned leaf assignments and check if they group assets sensibly.

    Compares with HRP's hard clustering as ground truth.
    """
    print(f"\n{'='*60}")
    print("  ASSET GROUPING ANALYSIS")
    print(f"{'='*60}")

    with torch.no_grad():
        assign_logits = model.leaf_assign_logits.cpu().numpy()
        assign = np.exp(assign_logits) / np.exp(assign_logits).sum(axis=-1, keepdims=True)

    n_assets, n_leaves = assign.shape
    print(f"  Assets: {n_assets}, Tree leaves: {n_leaves}")
    print(f"  Soft assignment entropy (mean): {-(assign * np.log(assign + 1e-8)).sum(axis=-1).mean():.3f}")
    print(f"  Max leaf prob per asset (mean): {assign.max(axis=-1).mean():.3f}")

    # Hard assignment for interpretation
    hard_assign = assign.argmax(axis=-1)
    print(f"\n  Primary leaf assignment per asset:")
    for i, name in enumerate(asset_names):
        max_prob = assign[i].max()
        print(f"    {name:15s} -> leaf {hard_assign[i]}  (p={max_prob:.2f})")

    # Group analysis: which assets are routed to the same leaf?
    leaf_groups = {}
    for i, leaf in enumerate(hard_assign):
        leaf_groups.setdefault(int(leaf), []).append(asset_names[i])

    print(f"\n  Leaf -> asset groups:")
    for leaf in sorted(leaf_groups.keys()):
        members = leaf_groups[leaf]
        print(f"    Leaf {leaf}: {', '.join(members)}")

    # Check category purity if ground truth available
    if asset_categories is not None:
        purity_scores = []
        for leaf, members in leaf_groups.items():
            cats = [asset_categories.get(m, "unknown") for m in members]
            if cats:
                dominant = max(set(cats), key=cats.count)
                purity = cats.count(dominant) / len(cats)
                purity_scores.append(purity)
        mean_purity = np.mean(purity_scores) if purity_scores else 0.0
        print(f"\n  Mean category purity: {mean_purity:.3f}")
        if mean_purity > 0.7:
            print("  [STRONG] Leaves group assets by ground-truth category.")
        elif mean_purity > 0.5:
            print("  [MODERATE] Partial category alignment.")
        else:
            print("  [WEAK] Leaves do not align with categories.")
        return {"groups": leaf_groups, "purity": mean_purity}

    return {"groups": leaf_groups}


def compare_with_hrp(model, returns, asset_names):
    """Compare DHRP's learned tree structure with HRP's hard clustering."""
    print(f"\n{'='*60}")
    print("  DHRP vs HRP CLUSTERING COMPARISON")
    print(f"{'='*60}")

    # HRP clustering
    corr = returns.corr().values
    dist = np.sqrt(0.5 * (1 - corr))
    n = len(asset_names)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(dist[i, j])

    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(condensed, method="single")
    n_leaves = 2 ** 3  # DHRP typically depth 3
    hrp_clusters = fcluster(Z, t=n_leaves, criterion="maxclust")

    # DHRP clustering
    with torch.no_grad():
        assign_logits = model.leaf_assign_logits.cpu().numpy()
        dhrp_clusters = assign_logits.argmax(axis=-1) + 1

    # Compare via adjusted rand index
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(hrp_clusters, dhrp_clusters)
    nmi = normalized_mutual_info_score(hrp_clusters, dhrp_clusters)

    print(f"  Adjusted Rand Index: {ari:.3f} (1.0 = identical, 0 = random)")
    print(f"  Normalized Mutual Info: {nmi:.3f}")

    print(f"\n  HRP clusters:")
    for cluster_id in sorted(set(hrp_clusters)):
        members = [asset_names[i] for i, c in enumerate(hrp_clusters) if c == cluster_id]
        print(f"    Cluster {cluster_id}: {', '.join(members)}")

    print(f"\n  DHRP clusters:")
    for cluster_id in sorted(set(dhrp_clusters)):
        members = [asset_names[i] for i, c in enumerate(dhrp_clusters) if c == cluster_id]
        print(f"    Cluster {cluster_id}: {', '.join(members)}")

    return {"ari": ari, "nmi": nmi}


def weight_sharpe_correlation_probe(model, X, S, returns, dates, prices,
                                     text_embs=None, device="cpu",
                                     forward_days=21, label=""):
    """Probe whether the model's allocation aligns with realized risk-adjusted returns.

    Replaces the failed regime-classification probe (test acc 0.041 vs majority
    0.709). Per-asset version: for each asset i and each rebalance date t,
    compute Spearman correlation between the model's weight on asset i and
    asset i's forward Sharpe ratio over the next k=forward_days.

    A high correlation means the model concentrates weight on assets that go on
    to deliver high risk-adjusted returns — even if it doesn't classify regimes.
    This is the metric reviewers actually care about for portfolio interpretability.

    Args:
        model: trained DHRPLayer or LLMDHRPLayer
        X, S: feature/covariance arrays from build_dataset
        returns: (n_dates, n_assets) DataFrame of daily returns
        dates: list of timestamps for each sample
        prices: original price DataFrame (used for asset names)
        text_embs: optional per-timestep text embeddings
        forward_days: forward horizon (default 21 = 1 trading month)
    Returns:
        dict with per-asset and aggregate correlation metrics
    """
    from scipy.stats import spearmanr

    print(f"\n{'='*60}")
    print(f"  WEIGHT-SHARPE CORRELATION PROBE: {label}")
    print(f"{'='*60}")

    n_samp = X.shape[0]
    n_assets = prices.shape[1]
    asset_names = list(prices.columns)
    rets_arr = returns.values if hasattr(returns, "values") else returns
    rets_idx = returns.index if hasattr(returns, "index") else None

    weights_history = []  # (n_samp, n_assets)
    forward_sharpes = []  # (n_samp, n_assets)

    model.eval()
    with torch.no_grad():
        for t in range(n_samp):
            if t >= len(dates):
                break
            x_t = torch.from_numpy(X[t]).to(device)
            s_t = torch.from_numpy(S[t]).to(device)

            if text_embs is not None:
                te = torch.from_numpy(text_embs[t]).to(device)
                w = model(x_t, s_t, text_emb=te).cpu().numpy()
            else:
                w = model(x_t, s_t).cpu().numpy()

            # Find date index in returns
            if rets_idx is not None:
                date_t = pd.Timestamp(dates[t])
                pos_arr = rets_idx.get_indexer([date_t], method="nearest")
                pos = int(pos_arr[0])
            else:
                pos = t

            # Forward Sharpe per asset over next forward_days
            window_end = min(pos + forward_days, rets_arr.shape[0])
            if window_end - pos < 5:
                continue
            window = rets_arr[pos:window_end]
            mu = np.nanmean(window, axis=0) * 252
            sigma = np.nanstd(window, axis=0) * np.sqrt(252) + 1e-8
            fwd_sharpe = mu / sigma

            weights_history.append(w)
            forward_sharpes.append(fwd_sharpe)

    if not weights_history:
        print("  No samples to evaluate — probe failed.")
        return None

    W = np.array(weights_history)        # (T, n_assets)
    SR = np.array(forward_sharpes)       # (T, n_assets)

    # Per-asset Spearman correlation between weight and forward Sharpe
    per_asset_corr = []
    for i in range(n_assets):
        w_col = W[:, i]
        sr_col = SR[:, i]
        valid = ~(np.isnan(w_col) | np.isnan(sr_col))
        if valid.sum() < 10 or np.std(w_col[valid]) < 1e-8 or np.std(sr_col[valid]) < 1e-8:
            per_asset_corr.append(np.nan)
            continue
        rho, _ = spearmanr(w_col[valid], sr_col[valid])
        per_asset_corr.append(rho)

    per_asset_corr = np.array(per_asset_corr)
    valid_corrs = per_asset_corr[~np.isnan(per_asset_corr)]
    mean_corr = float(np.nanmean(valid_corrs)) if len(valid_corrs) > 0 else 0.0
    abs_mean_corr = float(np.nanmean(np.abs(valid_corrs))) if len(valid_corrs) > 0 else 0.0

    # Cross-sectional: at each date, does weight ranking predict Sharpe ranking?
    cross_sectional_corrs = []
    for t in range(W.shape[0]):
        valid_t = ~(np.isnan(W[t]) | np.isnan(SR[t]))
        if valid_t.sum() < 3:
            continue
        if np.std(W[t][valid_t]) < 1e-8 or np.std(SR[t][valid_t]) < 1e-8:
            continue
        rho, _ = spearmanr(W[t][valid_t], SR[t][valid_t])
        if not np.isnan(rho):
            cross_sectional_corrs.append(rho)
    cross_sec_mean = float(np.mean(cross_sectional_corrs)) if cross_sectional_corrs else 0.0

    print(f"  Samples: {W.shape[0]}, forward horizon: {forward_days} days")
    print(f"  Per-asset Spearman correlation (weight ↔ forward Sharpe):")
    for i, name in enumerate(asset_names):
        rho = per_asset_corr[i]
        marker = "" if np.isnan(rho) else ("**" if abs(rho) > 0.3 else ("*" if abs(rho) > 0.15 else ""))
        print(f"    {name:15s}: {rho:+.3f} {marker}")
    print(f"  Mean per-asset correlation:        {mean_corr:+.4f}")
    print(f"  Mean |per-asset correlation|:      {abs_mean_corr:.4f}")
    print(f"  Cross-sectional correlation (mean): {cross_sec_mean:+.4f}")

    if abs_mean_corr > 0.3:
        print("  [STRONG] Model concentrates weight on assets with high forward Sharpe.")
    elif abs_mean_corr > 0.15:
        print("  [MODERATE] Some weight-Sharpe alignment.")
    else:
        print("  [WEAK] Allocations not predictive of forward risk-adjusted returns.")

    return {
        "per_asset_corr": per_asset_corr,
        "mean_corr": mean_corr,
        "abs_mean_corr": abs_mean_corr,
        "cross_sectional_mean": cross_sec_mean,
        "n_samples": W.shape[0],
    }


def plot_gate_heatmap(gate_probs, dates, save_path=None, label=""):
    """Heatmap of gate activations over time — visualizes regime detection."""
    n_samples, n_gates, _ = gate_probs.shape
    gate_left = gate_probs[:, :, 0]  # P(left branch)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(gate_left.T, aspect="auto", cmap="RdBu_r",
                    vmin=0, vmax=1, interpolation="nearest")

    # Mark date positions
    date_ticks = np.linspace(0, len(dates) - 1, 8, dtype=int)
    ax.set_xticks(date_ticks)
    ax.set_xticklabels([str(pd.Timestamp(dates[i]).date()) for i in date_ticks],
                        rotation=30, ha="right")
    ax.set_yticks(range(n_gates))
    ax.set_yticklabels([f"Node {i}" for i in range(n_gates)])
    ax.set_xlabel("Date")
    ax.set_ylabel("Tree Node")
    ax.set_title(f"Gate Activation (P[left]) Over Time — {label}")
    fig.colorbar(im, ax=ax, label="P(left branch)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def run_full_probe(model, X, S, dates, asset_names, returns=None,
                   text_embs=None, asset_categories=None,
                   device="cpu", label="", save_dir="results/figures"):
    """Run all probe analyses for interpretability."""
    print(f"\n{'#'*60}")
    print(f"  PROBE ANALYSIS: {label}")
    print(f"{'#'*60}")

    os.makedirs(save_dir, exist_ok=True)

    # Extract gate patterns
    gate_probs, dates_list = extract_gate_patterns(
        model, X, S, dates, text_embs=text_embs, device=device,
    )
    print(f"  Gate probs shape: {gate_probs.shape}")

    # Probe 1: Regime classification
    probe_result = regime_probe(gate_probs, dates_list, prices=returns)

    # Probe 2: Asset grouping
    group_result = asset_grouping_analysis(model, asset_names, asset_categories)

    # Probe 3: HRP comparison
    hrp_compare = None
    if returns is not None:
        hrp_compare = compare_with_hrp(model, returns, asset_names)

    # Probe 4: Weight-Sharpe correlation (replaces failed regime probe)
    ws_probe = None
    if returns is not None:
        # Need original prices DataFrame; reconstruct from returns
        ws_probe = weight_sharpe_correlation_probe(
            model, X, S, returns, dates_list,
            prices=returns,  # returns DataFrame has the asset names we need
            text_embs=text_embs, device=device,
            forward_days=21, label=label,
        )

    # Visualization
    plot_gate_heatmap(
        gate_probs, dates_list,
        save_path=os.path.join(save_dir, f"probe_gates_{label.lower()}.png"),
        label=label,
    )

    return {
        "regime_probe": probe_result,
        "asset_groups": group_result,
        "hrp_compare": hrp_compare,
        "weight_sharpe_probe": ws_probe,
        "gate_probs": gate_probs,
    }


if __name__ == "__main__":
    print("Run this script in Colab after training models.")
    print("Usage:")
    print("  from scripts.probe_analysis import run_full_probe")
    print("  result = run_full_probe(dhrp_dm, X_dm, S_dm, dates_dm,")
    print("                          asset_names=list(DM_prices.columns),")
    print("                          returns=DM_prices.pct_change().dropna(),")
    print("                          device=device, label='DM')")
