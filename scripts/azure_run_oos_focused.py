"""Focused OOS re-run: ablation studies + multi-seed robustness.

Re-runs ONLY the cells that had the OOS-leak bug (notebook cells 16 and 17),
using the strict held-out OOS protocol (TRAIN_END=2020-06-30, OOS_START=2020-07-01).

Skips: FinBERT/Qwen3 inference, LLM-DHRP training, deep baselines, main backtest,
       statistical battery — those cells already produced correct OOS numbers
       in the prior run and don't need to re-run.

Outputs:
  outputs/DM_ablations.csv         - tree depth + loss component ablations
  outputs/DM_multiseed.csv         - 5-seed multi-seed robustness summary
  outputs/DM_multiseed_per_seed.csv - per-seed metrics for boxplot
  outputs/HEAD_COMMIT.txt          - git commit hash for provenance

This is what the Azure ML job runs on a CPU-only Standard_E8s_v3 node.
Designed to be self-contained: it does not depend on prior cell state from
a notebook session.
"""
from __future__ import annotations
import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from src.data.price_loader import load_universe
from src.data.feature_engineering import build_dataset
from src.training.trainer import train_dhrp, train_dhrp_multiseed
from src.evaluation.backtest import rolling_backtest, multiseed_backtest
from src.evaluation.statistics import compute_stats
from src.models.dhrp_layer import DHRPLayer
from src.models.loss_functions import dhrp_loss

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TRAIN_END = "2020-06-30"
OOS_START = "2020-07-01"
END = datetime.now().strftime("%Y-%m-%d")
START = (datetime.now() - timedelta(days=14 * 365)).strftime("%Y-%m-%d")

# Always write to ./outputs in the current working directory.
# Azure ML auto-uploads ./outputs from the working directory at job completion.
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, os.cpu_count() or 1))


def log(msg: str) -> None:
    print(f"[run] {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


def write_provenance() -> None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unknown"
    (OUTPUTS / "HEAD_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (OUTPUTS / "run_metadata.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "device": device,
                "torch_version": torch.__version__,
                "cpu_count": os.cpu_count(),
                "torch_threads": torch.get_num_threads(),
                "start": START,
                "end": END,
                "train_end": TRAIN_END,
                "oos_start": OOS_START,
                "started_at": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"provenance: commit={commit[:12]}, device={device}, threads={torch.get_num_threads()}")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_data():
    log(f"Loading DM prices: {START} to {END}")
    DM_prices = load_universe("DM", START, END)
    log(f"  shape={DM_prices.shape}, dates={DM_prices.index[0].date()} to {DM_prices.index[-1].date()}")
    return DM_prices


# -----------------------------------------------------------------------------
# Ablation: tree depth + loss components
# -----------------------------------------------------------------------------
def train_ablation_variant(
    DM_prices,
    Xt, St, Rt, H,
    n_assets,
    label: str,
    config: str,
    *,
    depth: int = 3,
    use_crra: bool = True,
    use_sharpe: bool = True,
    use_hrp_reg: bool = True,
    epochs: int = 40,
):
    """Train one DHRP ablation variant and backtest it OOS."""
    fdim = Xt.shape[1]
    model = DHRPLayer(n_assets, fdim, hidden_dim=64, depth=depth, is_em=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4.5e-4, weight_decay=3e-4)

    best_l, best_s = float("inf"), None
    for ep in range(epochs):
        perm = torch.randperm(Xt.shape[0])
        el, nb_ = 0.0, 0
        lam = 0.3 - 0.2 * (ep / epochs)
        for s in range(0, Xt.shape[0], 32):
            e = min(s + 32, Xt.shape[0])
            opt.zero_grad()
            xb = Xt[perm[s:e]]
            Sb = St[perm[s:e]]
            Rb = Rt[perm[s:e]]
            Hb = H[perm[s:e].cpu().numpy()]
            if use_crra and use_sharpe and use_hrp_reg:
                # Full baseline path uses dhrp_loss
                loss = dhrp_loss(model, xb, Sb, Rb, Hb, is_em=False, lam_hrp=lam)
            else:
                # Custom loss without one of the components
                port_r, wts = [], []
                for t in range(Rb.shape[0]):
                    w = model(xb[t], Sb[t])
                    wts.append(w)
                    port_r.append((w * Rb[t]).sum())
                port_r = torch.stack(port_r)
                wts = torch.stack(wts)
                terms = []
                if use_crra:
                    crra = ((torch.clamp(1 + port_r, min=0.1) ** (1 - 2.5) - 1) / (1 - 2.5)).mean()
                    terms.append(-crra)
                if use_sharpe:
                    sharpe = port_r.mean() / (port_r.std() + 1e-6)
                    terms.append(-sharpe)
                if use_hrp_reg:
                    hrp_target = torch.from_numpy(Hb).to(device).float()
                    terms.append(((wts - hrp_target) ** 2).mean() * lam * 0.2)
                risk = torch.stack(
                    [wts[t] @ Sb[t] @ wts[t] for t in range(port_r.shape[0])]
                ).mean() * 0.001
                hhi = (wts ** 2).sum(1).mean() * 0.1
                terms.append(risk)
                terms.append(hhi)
                loss = sum(terms)
            if torch.isfinite(loss) and loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                el += loss.item()
                nb_ += 1
        if nb_ > 0 and el / nb_ < best_l:
            best_l = el / nb_
            best_s = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_s:
        model.load_state_dict({k: v.to(device) for k, v in best_s.items()})

    res = rolling_backtest(
        DM_prices,
        is_em=False,
        dhrp_model=model,
        methods=["DHRP"],
        oos_start=OOS_START,  # ← THE BUG FIX
    )
    stats = compute_stats(res)
    if stats.empty:
        log(f"  WARN: empty stats for {label}/{config}")
        return None
    row = stats.iloc[0].to_dict()
    row["Ablation"] = label
    row["Config"] = config
    log(f"  {label}[{config}]: Sharpe={row['Sharpe']:.4f}, MaxDD={row['MaxDD']:.4f}")
    return row


def run_ablation(DM_prices) -> pd.DataFrame:
    log("=== ABLATION (DM, OOS-corrected) ===")
    X, S, R, H = build_dataset(DM_prices, train_end=TRAIN_END)
    log(f"  in-sample dataset: X={X.shape}, S={S.shape}, R={R.shape}")
    n_assets = DM_prices.shape[1]

    Xt = torch.from_numpy(X).to(device)
    St = torch.from_numpy(S).to(device)
    Rt = torch.from_numpy(R).to(device)

    rows = []

    log("--- Tree depth ---")
    for depth in [2, 3, 4]:
        row = train_ablation_variant(
            DM_prices, Xt, St, Rt, H, n_assets,
            label="Tree depth", config=f"depth={depth}",
            depth=depth,
        )
        if row:
            rows.append(row)

    log("--- Loss components ---")
    for label, cfg, kwargs in [
        ("Loss component", "Full (baseline)", {}),
        ("Loss component", "No HRP reg", {"use_hrp_reg": False}),
        ("Loss component", "No Sharpe", {"use_sharpe": False}),
        ("Loss component", "No CRRA", {"use_crra": False}),
    ]:
        row = train_ablation_variant(
            DM_prices, Xt, St, Rt, H, n_assets,
            label=label, config=cfg, **kwargs,
        )
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = OUTPUTS / "DM_ablations.csv"
    df.to_csv(out_path, index=False)
    log(f"Saved {out_path}")
    return df


# -----------------------------------------------------------------------------
# Multi-seed robustness
# -----------------------------------------------------------------------------
def run_multiseed(DM_prices) -> pd.DataFrame:
    log("=== MULTI-SEED (DM, 5 seeds, OOS-corrected) ===")
    seeds = [0, 1, 2, 3, 4]
    dhrp_models = train_dhrp_multiseed(
        DM_prices,
        seeds=seeds,
        device=device,
        is_em=False,
        train_end=TRAIN_END,
    )
    log(f"  trained {len(dhrp_models)} models")

    models_by_seed = [{"dhrp": m} for m in dhrp_models]
    seed_agg = multiseed_backtest(
        DM_prices,
        models_by_seed,
        is_em=False,
        methods=["EW", "MINVAR", "MV", "HRP", "RP", "MAXDIV", "DHRP"],
        oos_start=OOS_START,  # ← THE BUG FIX
    )
    out_path = OUTPUTS / "DM_multiseed.csv"
    seed_agg.to_csv(out_path, index=False)
    log(f"Saved {out_path}")

    # Per-seed table for boxplot
    log("--- Per-seed OOS backtest ---")
    per_seed_rows = []
    for i, m in enumerate(dhrp_models):
        res_i = rolling_backtest(
            DM_prices,
            is_em=False,
            dhrp_model=m,
            methods=["EW", "MINVAR", "MV", "HRP", "RP", "MAXDIV", "DHRP"],
            oos_start=OOS_START,
        )
        s = compute_stats(res_i)
        s["seed"] = i
        per_seed_rows.append(s)
        dhrp_row = s[s["Method"] == "DHRP"]
        if not dhrp_row.empty:
            log(f"  seed {i}: DHRP Sharpe={dhrp_row.iloc[0]['Sharpe']:.4f}")
    per_seed = pd.concat(per_seed_rows, ignore_index=True)
    per_seed_path = OUTPUTS / "DM_multiseed_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)
    log(f"Saved {per_seed_path}")

    return seed_agg


# -----------------------------------------------------------------------------
# Sanity check: re-derive cell 11's DHRP DM Sharpe so we can compare consistency
# -----------------------------------------------------------------------------
def run_sanity_check(DM_prices) -> dict:
    log("=== SANITY: train one DHRP at default settings, backtest OOS ===")
    log("(should match cell 11 within seed noise)")
    model = train_dhrp(
        DM_prices,
        device=device,
        is_em=False,
        train_end=TRAIN_END,
        seed=42,
    )
    res = rolling_backtest(
        DM_prices,
        is_em=False,
        dhrp_model=model,
        methods=["DHRP", "EW", "HRP", "MV"],
        oos_start=OOS_START,
    )
    stats = compute_stats(res)
    out = {row["Method"]: float(row["Sharpe"]) for _, row in stats.iterrows()}
    log(f"  sanity Sharpes: {out}")
    (OUTPUTS / "sanity_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main():
    log("=" * 70)
    log("DHRP focused OOS re-run (ablation + multi-seed only)")
    log(f"  device          = {device}")
    log(f"  torch_threads   = {torch.get_num_threads()}")
    log(f"  data window     = {START} to {END}")
    log(f"  train_end       = {TRAIN_END}")
    log(f"  oos_start       = {OOS_START}")
    log(f"  output dir      = {OUTPUTS.absolute()}")
    log("=" * 70)

    write_provenance()

    t0 = time.time()
    DM_prices = load_data()

    sanity = run_sanity_check(DM_prices)
    ablation_df = run_ablation(DM_prices)
    multiseed_df = run_multiseed(DM_prices)

    elapsed = time.time() - t0
    log(f"All done in {elapsed/60:.1f} minutes")
    log(f"Sanity DHRP DM Sharpe = {sanity.get('DHRP', 'N/A'):.4f}")
    if ablation_df is not None and not ablation_df.empty:
        full_baseline = ablation_df[
            (ablation_df["Ablation"] == "Loss component")
            & (ablation_df["Config"] == "Full (baseline)")
        ]
        if not full_baseline.empty:
            log(f"Ablation full-baseline Sharpe = {full_baseline.iloc[0]['Sharpe']:.4f}")
    if multiseed_df is not None and not multiseed_df.empty:
        dhrp_row = multiseed_df[multiseed_df["Method"] == "DHRP"]
        if not dhrp_row.empty:
            log(
                f"Multi-seed DHRP Sharpe mean = {dhrp_row.iloc[0]['Sharpe_mean']:.4f} "
                f"+/- {dhrp_row.iloc[0]['Sharpe_std']:.4f}"
            )


if __name__ == "__main__":
    main()
