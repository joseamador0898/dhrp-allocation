#!/usr/bin/env python3
"""Run the full ablation study matrix (A1-A12).

Ablation IDs:
    A1:  Price + DHRP (no LLM)             — original baseline
    A2:  Price + Sentiment + DHRP          — structured LLM output only
    A3:  Price + Embeddings + DHRP         — dense representations only
    A4:  Price + Both LLM + DHRP           — full model
    A5:  Price + Both LLM + MLP            — flat architecture (no tree)
    A6:  LLM only + DHRP (no price)        — how much do price features add?
    A7:  Price + Both LLM + DHRP depth=1   — shallow tree
    A8:  Price + Both LLM + DHRP depth=5   — deep tree
    A9:  Price + Both LLM + DHRP no HRP    — value of HRP regularization
    A10: DHRP + volume/IV features         — market microstructure value
    A11: Transformer + LLM features        — attention vs tree for LLM fusion
    A12: PPO + price features              — RL vs differentiable optimization
"""

import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

from src.data.price_loader import load_universe, load_etf_volume_data, UNIVERSES
from src.data.feature_engineering import build_dataset, DEFAULT_FDIM
from src.training.trainer import train_dhrp, train_llm_dhrp
from src.models.deep_baselines import (
    train_transformer_policy, train_ppo_agent,
)
from src.evaluation.backtest import rolling_backtest
from src.evaluation.statistics import compute_stats, sharpe_difference_test, fdr_correct

ABLATIONS = {
    "A1":  {"name": "DHRP (price-only)", "model": "dhrp", "use_text": False, "use_macro": False, "use_volume": False},
    "A2":  {"name": "DHRP + sentiment", "model": "llm_dhrp", "use_text": True, "text_type": "sentiment"},
    "A3":  {"name": "DHRP + embeddings", "model": "llm_dhrp", "use_text": True, "text_type": "finbert"},
    "A4":  {"name": "DHRP + both (full)", "model": "llm_dhrp", "use_text": True, "text_type": "both"},
    "A5":  {"name": "MLP + both (flat)", "model": "mlp", "use_text": True},
    "A6":  {"name": "DHRP + text, no price", "model": "llm_dhrp", "use_text": True, "zero_price": True},
    "A7":  {"name": "DHRP + both, depth=1", "model": "llm_dhrp", "use_text": True, "depth": 1},
    "A8":  {"name": "DHRP + both, depth=5", "model": "llm_dhrp", "use_text": True, "depth": 5},
    "A9":  {"name": "DHRP + both, no HRP reg", "model": "llm_dhrp", "use_text": True, "hrp_reg": False},
    "A10": {"name": "DHRP + volume features", "model": "dhrp", "use_text": False, "use_volume": True},
    "A11": {"name": "Transformer + LLM", "model": "transformer", "use_text": True},
    "A12": {"name": "PPO agent", "model": "ppo", "use_text": False},
}


def run_ablation_suite(
    universe_name="Commodities",
    device="cpu",
    text_features=None,
    macro_features=None,
    ablation_ids=None,
    output_dir="results/ablations",
):
    """Run full ablation suite for a single universe.

    Args:
        universe_name: "DM", "EM", or "Commodities"
        device: compute device
        text_features: dict with 'finbert' array, or None
        macro_features: np.ndarray, or None
        ablation_ids: list of ablation IDs to run, or None for all
        output_dir: output directory
    Returns:
        DataFrame with results for all ablations
    """
    if ablation_ids is None:
        ablation_ids = list(ABLATIONS.keys())

    os.makedirs(output_dir, exist_ok=True)
    is_em = universe_name == "EM"
    fdim = DEFAULT_FDIM

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=10 * 365)).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  ABLATION SUITE: {universe_name}")
    print(f"{'='*60}")

    prices = load_universe(universe_name, start, end)
    n_assets = prices.shape[1]

    # Load volume data
    volume = None
    try:
        volume = load_etf_volume_data(UNIVERSES[universe_name], start, end)
    except Exception:
        print("  Volume data not available.")

    results = []
    backtest_results = {}

    for abl_id in ablation_ids:
        if abl_id not in ABLATIONS:
            continue
        cfg = ABLATIONS[abl_id]
        print(f"\n--- {abl_id}: {cfg['name']} ---")

        try:
            model = None
            vol = volume if cfg.get("use_volume", False) else None
            use_macro = cfg.get("use_macro", True) and macro_features is not None

            if cfg["model"] == "dhrp":
                model = train_dhrp(prices, device=device, is_em=is_em, volume=vol, fdim=fdim)

            elif cfg["model"] == "llm_dhrp":
                depth = cfg.get("depth", 3)
                use_hrp = cfg.get("hrp_reg", True)
                model = train_llm_dhrp(
                    prices, text_features=text_features,
                    macro_features=macro_features,
                    device=device, is_em=is_em, volume=vol, fdim=fdim,
                    use_text=cfg.get("use_text", True),
                    use_macro=use_macro, depth=depth,
                    use_hrp_reg=use_hrp,
                    fusion_type="cross_attention",
                    epochs=50, lr=3e-4,
                )

            elif cfg["model"] == "transformer":
                model = train_transformer_policy(
                    prices, device=device, is_em=is_em, volume=vol, fdim=fdim,
                    epochs=40, lr=3e-4,
                )

            elif cfg["model"] == "ppo":
                model = train_ppo_agent(
                    prices, device=device, is_em=is_em, volume=vol, fdim=fdim,
                    epochs=30, lr=3e-4,
                )

            elif cfg["model"] == "mlp":
                # Train MLP via same loss as DHRP
                from src.models.deep_baselines import MLPWithCovPolicy
                from src.models.loss_functions import dhrp_loss

                X, S, R, H = build_dataset(prices, is_em=is_em, volume=vol, fdim=fdim)
                model = MLPWithCovPolicy(X.shape[1], n_assets).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-4)
                Xt = torch.from_numpy(X).to(device)
                St = torch.from_numpy(S).to(device)
                Rt = torch.from_numpy(R).to(device)

                best_loss, best_st = float("inf"), None
                for ep in range(40):
                    perm = torch.randperm(X.shape[0])
                    ep_loss, nb = 0.0, 0
                    for s in range(0, X.shape[0], 32):
                        e = min(s + 32, X.shape[0])
                        opt.zero_grad()
                        loss = dhrp_loss(model, Xt[perm[s:e]], St[perm[s:e]], Rt[perm[s:e]],
                                         H[perm[s:e].cpu().numpy()], is_em=is_em)
                        if not torch.isnan(loss) and loss.requires_grad:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            opt.step()
                            ep_loss += loss.item()
                            nb += 1
                    if nb > 0:
                        avg = ep_loss / nb
                        if avg < best_loss:
                            best_loss = avg
                            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        if (ep + 1) % 10 == 0:
                            print(f"  [MLP] Epoch {ep+1}/40, loss={avg:.6f}")
                if best_st:
                    model.load_state_dict({k: v.to(device) for k, v in best_st.items()})

            n_params = sum(p.numel() for p in model.parameters()) if model else 0
            print(f"  Params: {n_params:,}")

            # Backtest
            methods = ["EW", "HRP", "DHRP"] if model else ["EW", "HRP"]
            bt = rolling_backtest(
                prices, is_em=is_em, dhrp_model=model,
                methods=["DHRP"], volume=vol,
            )
            # Rename DHRP to the ablation ID for tracking
            bt["method"] = abl_id
            backtest_results[abl_id] = bt

            # Compute stats
            sharpe = compute_stats(bt).iloc[0]["Sharpe"] if not bt.empty else np.nan
            results.append({
                "Ablation": abl_id, "Name": cfg["name"],
                "Sharpe": sharpe, "Params": n_params, "N_obs": len(bt),
            })
            print(f"  Sharpe: {sharpe:.3f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"Ablation": abl_id, "Name": cfg["name"], "Sharpe": np.nan, "Error": str(e)})

    # Summary
    summary = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"  ABLATION RESULTS: {universe_name}")
    print(f"{'='*60}")
    print(summary.to_string(index=False))

    # FDR correction on pairwise tests
    if "A1" in backtest_results and len(backtest_results) > 1:
        print(f"\n--- Pairwise tests vs A1 (DHRP baseline) ---")
        base = backtest_results["A1"].copy()
        base["method"] = "A1"
        p_values = []
        test_names = []
        for abl_id, bt in backtest_results.items():
            if abl_id == "A1":
                continue
            try:
                combined = pd.concat([base, bt], ignore_index=True)
                diff = sharpe_difference_test(combined, abl_id, "A1")
                p_values.append(diff["bootstrap_p"])
                test_names.append(abl_id)
                print(f"  {abl_id} vs A1: diff={diff['sharpe_diff']:+.3f} p={diff['bootstrap_p']:.3f}")
            except Exception:
                pass

        if p_values:
            fdr = fdr_correct(p_values)
            print(f"\n  FDR-adjusted p-values (Benjamini-Hochberg):")
            for name, raw, adj in zip(test_names, fdr["raw"], fdr["adjusted"]):
                sig = " *" if adj < 0.10 else ""
                print(f"    {name}: raw={raw:.3f} adj={adj:.3f}{sig}")

    summary.to_csv(f"{output_dir}/{universe_name}_ablations.csv", index=False)
    print(f"\nSaved to {output_dir}/{universe_name}_ablations.csv")
    return summary


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load precomputed text features if available
    text_features = None
    for universe in ["Commodities", "DM", "EM"]:
        text_path = f"results/features/text_{universe.lower()[:3]}.npz"
        if os.path.exists(text_path):
            data = np.load(text_path)
            text_features = {"finbert": data["finbert"]}
            print(f"Loaded text features: {text_path}")

        run_ablation_suite(
            universe_name=universe, device=device,
            text_features=text_features,
            ablation_ids=["A1", "A4", "A10", "A11", "A12"],  # Quick subset
        )
