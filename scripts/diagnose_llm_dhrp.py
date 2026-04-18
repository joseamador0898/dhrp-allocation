"""Diagnostic script for LLM-DHRP underperformance.

Investigates:
1. Text gate values — are gates staying near zero (text suppressed)?
2. Routing shifts — does text actually change tree routing decisions?
3. Text feature alignment — are FinBERT embeddings aligned to correct dates?
4. FinBERT embedding quality — do embeddings separate across market regimes?

Run in Colab after Cell 8 (models trained) or load saved models.
"""

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def diagnose_gate_values(model, X, S, text_embs, device="cpu", label=""):
    """Extract text gate values across all samples to check if text is suppressed."""
    print(f"\n{'='*60}")
    print(f"  GATE VALUE ANALYSIS: {label}")
    print(f"{'='*60}")

    gate_means = []
    gate_stds = []
    n_samp = min(X.shape[0], text_embs.shape[0])

    model.eval()
    with torch.no_grad():
        for t in range(n_samp):
            x_t = torch.from_numpy(X[t]).to(device)
            s_t = torch.from_numpy(S[t]).to(device)
            te = torch.from_numpy(text_embs[t]).to(device)

            result = model.get_text_gate_values(x_t, s_t, text_emb=te)
            if result is not None:
                gate_means.append(result["gate_mean"])
                gate_stds.append(result["gate_std"])

    if gate_means:
        gate_means = np.array(gate_means)
        gate_stds = np.array(gate_stds)
        print(f"  Gate mean: {np.mean(gate_means):.4f} +/- {np.std(gate_means):.4f}")
        print(f"  Gate std:  {np.mean(gate_stds):.4f}")
        print(f"  Gate range: [{np.min(gate_means):.4f}, {np.max(gate_means):.4f}]")
        print(f"  % gates > 0.3: {100 * np.mean(gate_means > 0.3):.1f}%")
        print(f"  % gates > 0.5: {100 * np.mean(gate_means > 0.5):.1f}%")

        # Interpretation
        if np.mean(gate_means) < 0.15:
            print("  ** WARNING: Gates near zero — text is being heavily suppressed!")
            print("     The model is barely using text features.")
            print("     Consider: gate_bias_init=-1.0 (sigmoid=0.27) instead of -2.0 (0.12)")
        elif np.mean(gate_means) > 0.5:
            print("  ** Text is actively used — gate values are high")
        else:
            print("  Text influence is moderate")

        return gate_means
    else:
        print("  No gate values extracted (model may not use cross_attention fusion)")
        return None


def diagnose_routing_shifts(model, X, S, text_embs, device="cpu", label=""):
    """Check how much text changes tree routing decisions."""
    print(f"\n{'='*60}")
    print(f"  ROUTING SHIFT ANALYSIS: {label}")
    print(f"{'='*60}")

    total_shifts = []
    root_shifts = []
    n_samp = min(X.shape[0], text_embs.shape[0])

    model.eval()
    with torch.no_grad():
        for t in range(n_samp):
            x_t = torch.from_numpy(X[t]).to(device)
            s_t = torch.from_numpy(S[t]).to(device)
            te = torch.from_numpy(text_embs[t]).to(device)

            shifts = model.get_routing_shift(x_t, s_t, text_emb=te)
            total_shifts.append(shifts["total"])
            root_shifts.append(shifts["root"])

    total_shifts = np.array(total_shifts)
    root_shifts = np.array(root_shifts)

    print(f"  Total routing shift: {np.mean(total_shifts):.4f} +/- {np.std(total_shifts):.4f}")
    print(f"  Root node shift:     {np.mean(root_shifts):.4f} +/- {np.std(root_shifts):.4f}")
    print(f"  Max total shift:     {np.max(total_shifts):.4f}")
    print(f"  % shifts > 0.01:    {100 * np.mean(total_shifts > 0.01):.1f}%")
    print(f"  % shifts > 0.1:     {100 * np.mean(total_shifts > 0.1):.1f}%")

    if np.mean(total_shifts) < 0.01:
        print("  ** WARNING: Text has negligible effect on routing!")
        print("     The tree structure ignores text features entirely.")
    elif np.mean(total_shifts) > 0.5:
        print("  ** Text dramatically changes routing — possible instability")
    else:
        print("  Text modestly influences routing decisions")

    return total_shifts, root_shifts


def diagnose_weight_impact(model, X, S, text_embs, device="cpu", label=""):
    """Compare portfolio weights with and without text to measure actual impact."""
    print(f"\n{'='*60}")
    print(f"  WEIGHT IMPACT ANALYSIS: {label}")
    print(f"{'='*60}")

    weight_diffs = []
    n_samp = min(X.shape[0], text_embs.shape[0])

    model.eval()
    with torch.no_grad():
        for t in range(n_samp):
            x_t = torch.from_numpy(X[t]).to(device)
            s_t = torch.from_numpy(S[t]).to(device)
            te = torch.from_numpy(text_embs[t]).to(device)

            w_with = model(x_t, s_t, text_emb=te).cpu().numpy()
            w_without = model(x_t, s_t, text_emb=None).cpu().numpy()
            diff = np.sum(np.abs(w_with - w_without))
            weight_diffs.append(diff)

    weight_diffs = np.array(weight_diffs)
    print(f"  Mean L1 weight diff: {np.mean(weight_diffs):.4f}")
    print(f"  Max L1 weight diff:  {np.max(weight_diffs):.4f}")
    print(f"  % diffs > 0.01:     {100 * np.mean(weight_diffs > 0.01):.1f}%")
    print(f"  % diffs > 0.1:      {100 * np.mean(weight_diffs > 0.1):.1f}%")

    if np.mean(weight_diffs) < 0.01:
        print("  ** Text has essentially NO effect on final portfolio weights")
    elif np.mean(weight_diffs) > 0.2:
        print("  ** Text causes large weight swings — potential noise amplification")

    return weight_diffs


def diagnose_text_alignment(text_features, prices, train_end=None, label=""):
    """Check if text features are properly aligned with price data dates."""
    print(f"\n{'='*60}")
    print(f"  TEXT ALIGNMENT CHECK: {label}")
    print(f"{'='*60}")

    if text_features is None or "finbert" not in text_features:
        print("  No text features to check")
        return

    fb = text_features["finbert"]
    if isinstance(fb, dict):
        print(f"  Text features are date-keyed dict with {len(fb)} entries")
        dates = sorted(fb.keys())
        print(f"  Date range: {dates[0]} to {dates[-1]}")
        print(f"  Price range: {prices.index[0]} to {prices.index[-1]}")
        # Check overlap
        price_dates = set(prices.index)
        text_dates = set(dates)
        overlap = len(price_dates & text_dates)
        print(f"  Date overlap: {overlap}/{len(text_dates)} text dates match price dates")
    elif isinstance(fb, np.ndarray):
        print(f"  Text features are positional array: shape={fb.shape}")
        print(f"  Price data: {len(prices)} rows")
        from src.data.feature_engineering import build_dataset
        X, S, R, H = build_dataset(prices, train_end=train_end)
        print(f"  build_dataset samples: {X.shape[0]}")
        print(f"  Text samples: {fb.shape[0]}")
        if X.shape[0] != fb.shape[0]:
            print(f"  ** MISALIGNMENT: {X.shape[0]} price samples vs {fb.shape[0]} text samples!")
            print(f"     The text features will be truncated/padded, possibly misaligning dates")
        else:
            print(f"  Counts match — alignment appears correct")


def diagnose_embedding_quality(text_embs, returns=None, label=""):
    """Check if FinBERT embeddings have regime-discriminative structure."""
    print(f"\n{'='*60}")
    print(f"  EMBEDDING QUALITY CHECK: {label}")
    print(f"{'='*60}")

    if text_embs is None:
        print("  No text embeddings to analyze")
        return

    print(f"  Embedding shape: {text_embs.shape}")
    print(f"  Mean norm: {np.mean(np.linalg.norm(text_embs, axis=-1)):.2f}")
    print(f"  Std norm:  {np.std(np.linalg.norm(text_embs, axis=-1)):.2f}")

    # Check for zero/constant embeddings (padding)
    norms = np.linalg.norm(text_embs, axis=-1)
    pct_zero = 100 * np.mean(norms < 1e-6)
    print(f"  % zero embeddings: {pct_zero:.1f}%")

    if pct_zero > 20:
        print("  ** WARNING: >20% of embeddings are zero (padded) — text signal is sparse")

    # Check cosine similarity distribution
    n = min(100, len(text_embs))
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            if norms[i] > 1e-6 and norms[j] > 1e-6:
                sim = np.dot(text_embs[i], text_embs[j]) / (norms[i] * norms[j])
                sims.append(sim)

    if sims:
        sims = np.array(sims)
        print(f"  Cosine similarity: mean={np.mean(sims):.3f}, std={np.std(sims):.3f}")
        if np.std(sims) < 0.05:
            print("  ** WARNING: Embeddings are nearly identical — no discriminative signal!")
            print("     All headlines may be producing the same embedding")
        elif np.mean(sims) > 0.95:
            print("  ** WARNING: Very high average similarity — weak differentiation")

    # If returns provided, check if embeddings correlate with market regimes
    if returns is not None and len(returns) >= len(text_embs):
        n_emb = len(text_embs)
        market_ret = returns[:n_emb].mean(axis=1) if returns.ndim > 1 else returns[:n_emb]
        # Split into positive/negative return regimes
        pos_mask = market_ret > 0
        neg_mask = market_ret <= 0
        if pos_mask.sum() > 5 and neg_mask.sum() > 5:
            pos_norms = norms[pos_mask].mean()
            neg_norms = norms[neg_mask].mean()
            print(f"  Mean norm (positive returns): {pos_norms:.2f}")
            print(f"  Mean norm (negative returns): {neg_norms:.2f}")


def plot_diagnostics(gate_means, total_shifts, weight_diffs, save_path=None, label=""):
    """Create diagnostic plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"LLM-DHRP Diagnostics: {label}", fontsize=14)

    if gate_means is not None:
        axes[0].hist(gate_means, bins=50, alpha=0.7, color="steelblue")
        axes[0].axvline(np.mean(gate_means), color="red", linestyle="--",
                       label=f"mean={np.mean(gate_means):.3f}")
        axes[0].set_xlabel("Gate Value (sigmoid output)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Text Gate Distribution")
        axes[0].legend()

    if total_shifts is not None:
        axes[1].hist(total_shifts, bins=50, alpha=0.7, color="coral")
        axes[1].axvline(np.mean(total_shifts), color="red", linestyle="--",
                       label=f"mean={np.mean(total_shifts):.3f}")
        axes[1].set_xlabel("Total Routing Shift")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Routing Shift Distribution")
        axes[1].legend()

    if weight_diffs is not None:
        axes[2].hist(weight_diffs, bins=50, alpha=0.7, color="seagreen")
        axes[2].axvline(np.mean(weight_diffs), color="red", linestyle="--",
                       label=f"mean={np.mean(weight_diffs):.3f}")
        axes[2].set_xlabel("L1 Weight Difference")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Weight Impact Distribution")
        axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved diagnostic plot: {save_path}")
    plt.close()


def run_full_diagnostics(
    model, X, S, text_embs, text_features=None, prices=None,
    returns=None, device="cpu", label="", save_dir="results/figures",
    train_end=None,
):
    """Run all diagnostic checks on a trained LLM-DHRP model."""
    print(f"\n{'#'*60}")
    print(f"  FULL DIAGNOSTICS: {label}")
    print(f"{'#'*60}")

    # 1. Gate values
    gate_means = diagnose_gate_values(model, X, S, text_embs, device, label)

    # 2. Routing shifts
    total_shifts, root_shifts = diagnose_routing_shifts(model, X, S, text_embs, device, label)

    # 3. Weight impact
    weight_diffs = diagnose_weight_impact(model, X, S, text_embs, device, label)

    # 4. Text alignment
    if text_features is not None and prices is not None:
        diagnose_text_alignment(text_features, prices, train_end, label)

    # 5. Embedding quality
    diagnose_embedding_quality(text_embs, returns, label)

    # 6. Plot
    os.makedirs(save_dir, exist_ok=True)
    plot_diagnostics(
        gate_means, total_shifts, weight_diffs,
        save_path=os.path.join(save_dir, f"diagnostics_{label.lower().replace(' ', '_')}.png"),
        label=label,
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS SUMMARY: {label}")
    print(f"{'='*60}")
    issues = []
    if gate_means is not None and np.mean(gate_means) < 0.15:
        issues.append("Text gates suppressed (mean < 0.15)")
    if np.mean(total_shifts) < 0.01:
        issues.append("Routing shifts negligible (mean < 0.01)")
    if np.mean(weight_diffs) < 0.01:
        issues.append("Weight impact negligible (mean < 0.01)")

    # Embedding quality issues (BERT anisotropy) — propagate to summary
    if text_embs is not None and len(text_embs) >= 2:
        norms = np.linalg.norm(text_embs, axis=-1)
        good = norms > 1e-6
        if good.sum() >= 2:
            X = text_embs[good]
            sims = []
            n = min(50, len(X))
            for i in range(n):
                for j in range(i + 1, n):
                    s = np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]) + 1e-8)
                    sims.append(s)
            if sims:
                sims = np.array(sims)
                if np.std(sims) < 0.05 or np.mean(sims) > 0.95:
                    issues.append(
                        f"Embeddings anisotropic (cos sim mean={np.mean(sims):.3f}, "
                        f"std={np.std(sims):.3f}) — apply Soft-ZCA whitening or use "
                        f"FinLang/finance-embeddings-investopedia"
                    )

    if issues:
        print("  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
        print("\n  RECOMMENDED FIXES:")
        print("    1. Use gate_bias_init=-1.0 (sigmoid=0.27) instead of -2.0 (0.12)")
        print("    2. Increase text_lr_scale from 0.3 to 0.5")
        print("    3. Reduce modality_dropout from 0.2 to 0.1")
        print("    4. Try warm-start training: train_llm_dhrp_warmstart()")
        print("    5. Apply Soft-ZCA whitening (apply_soft_zca_whitening in llm_features.py)")
        print("    6. Or switch to FinLang/finance-embeddings-investopedia sentence-transformer")
    else:
        print("  No critical issues found — text pathway is active")
        print("  If performance is still poor, the issue may be text quality, not architecture")

    return {
        "gate_means": gate_means,
        "total_shifts": total_shifts,
        "root_shifts": root_shifts,
        "weight_diffs": weight_diffs,
        "issues": issues,
    }


if __name__ == "__main__":
    print("Run this script in Colab after training LLM-DHRP models.")
    print("Usage:")
    print("  from scripts.diagnose_llm_dhrp import run_full_diagnostics")
    print("  results = run_full_diagnostics(llm_dhrp_dm, X_dm, S_dm, text_embs_dm,")
    print("                                 device=device, label='DM')")
