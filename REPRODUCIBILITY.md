# Reproducibility Guide

This document describes how to reproduce all results in the paper
**"DHRP: A Differentiable Hierarchical Risk Parity Architecture and Multi-Universe Portfolio Benchmark"** (NeurIPS 2026 Evaluations & Datasets track).

## TL;DR

```bash
git clone https://github.com/<anonymous>/dhrp-allocation.git
cd dhrp-allocation
pip install -r requirements-frozen.txt
bash scripts/reproduce_all.sh    # ~10 GPU-hours on T4, ~3-5h on A100
```

All paper figures and tables will be generated in `results/figures/` and
`results/*.csv` respectively.

## Environment

- **Python**: 3.10+
- **PyTorch**: 2.1.0 with CUDA 12.1 (T4) or CUDA 11.8 (A100)
- **GPU**: Recommended T4 16GB or A100 40GB. CPU-only is feasible but slow (~20× slower).
- **OS**: Tested on Ubuntu 22.04 (Google Colab) and Windows 11

Pinned dependencies in `requirements-frozen.txt` (generated via `pip freeze` from the verified Colab Pro environment).

## Data sources

All data is publicly available; no API keys required for the headline experiments.

| Source | What we use | Access |
|--------|-------------|--------|
| Yahoo Finance v8 | ETF daily OHLCV for all 80 tickers | Public REST API, cached locally |
| Kenneth French Data Library | Fama-French 5 + Momentum factors | Public CSV download |
| AQR Capital Management | Commodity Value/Momentum factors | Public CSV download (research portal) |
| Reuters / Bloomberg / GDELT 2.0 | Financial news headlines | Public RSS feeds, GDELT BigQuery (free tier) |
| HuggingFace `ProsusAI/finbert` | FinBERT embeddings | Public model |

For full attribution and licensing details see `data/croissant/dhrp-8universe.json` (Croissant 1.1 metadata).

## Single-command reproduction

```bash
bash scripts/reproduce_all.sh
```

This script runs:
1. **Cell 1** (setup): Verify CUDA, install dependencies, mount Drive
2. **Cell 2** (price data): Download all 80 ETF tickers, cache locally
3. **Cell 3-4** (headlines + FinBERT): Extract headlines and FinBERT embeddings
4. **Cell 5** (Gemini sentiment): Optional — requires GOOGLE_API_KEY env var; skipped by default
5. **Cell 6** (text feature tensors): Build temporal per-asset text features (90-day rolling windows)
6. **Cell 7** (macro features): Download FRED + GS Quant + SPGCI macro features
7. **Cell 8-9** (training): Train DHRP and LLM-DHRP for DM/EM/Commodities (single seed for sanity)
8. **Cell 13-14** (deep baselines + full backtest): MLP, Transformer, PPO, DFL across all methods
9. **Cell 15-19** (statistical tests + ablations): Pairwise Sharpe, DM tests, ablations, multi-seed, cost sensitivity, regime
10. **Cell 23** (multi-seed expansion): 10 seeds × 8 universes (DHRP + LLM-DHRP) — the main paper table
11. **Cell 24** (probe analysis): Asset grouping + weight-Sharpe correlation interpretability

## Per-cell reproduction (Jupyter)

Open `notebooks/llm_dhrp_experiments.ipynb` in Jupyter or Colab and run cells sequentially. Each cell is idempotent and produces deterministic output for a given random seed.

## Random seeds

All neural methods are trained with seeds `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` (10 seeds). Set via `SEEDS = [0, 1, ..., 9]` in Cell 23. PyTorch deterministic mode is set via `torch.manual_seed(seed)` and `np.random.seed(seed)` at the top of each training function.

Classical baselines (EW, MV, MINVAR, HRP, RP, MAXDIV) are deterministic and produce identical output across runs.

## Compute budget

| Step | GPU-hours (T4) | GPU-hours (A100) |
|------|----------------|-------------------|
| FinBERT extraction (Cell 4) | 0.5 | 0.1 |
| Single-seed training (Cells 8-9, 13) | 0.5 | 0.1 |
| Multi-seed × 8 universes (Cell 23) | 6.0 | 1.5 |
| Statistical analysis (Cells 11, 14-19) | 0.2 (CPU) | — |
| Probe analysis (Cell 24) | 0.1 | 0.05 |
| **Total** | **~7.3 GPU-hours** | **~1.8 GPU-hours** |

Plus ~30 minutes CPU time for figure generation, table compilation, and Croissant validation.

## Verification checklist

After running `reproduce_all.sh`, verify:

- [ ] `results/sharpe_pivot_multiseed_mean.csv` shows DHRP Sharpe ≥ 0.95 in Commodities, ≥ 1.20 in Crypto
- [ ] `results/psr_pivot_multiseed.csv` shows DHRP PSR > 0.95 in Commodities and Crypto
- [ ] `results/Commodities_results.csv` shows DHRP FF3 alpha t-stat > 2.0
- [ ] All 8 universes have non-empty multi-seed result CSVs
- [ ] `results/figures/cumulative_returns.png` renders without errors
- [ ] No NaN errors in cell outputs

## Anonymized code release

For NeurIPS 2026 double-blind review, the code is hosted on `anonymous.4open.science` at the URL referenced in the paper. The same code is available on GitHub under the de-anonymized identifier post-acceptance.

## Contact

Anonymized during review. Post-acceptance: see paper for author email.
