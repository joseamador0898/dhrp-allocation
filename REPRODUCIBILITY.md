# Reproducibility Guide

This document describes how to reproduce all results in the paper
**"DHRP: A Differentiable Hierarchical Risk Parity Architecture and
Multi-Universe Portfolio Benchmark"** (NeurIPS 2026 Evaluations &
Datasets track).

## TL;DR

```bash
git clone <repository-url>
cd dhrp-allocation
pip install -r requirements.txt
python tests/test_imports.py            # smoke test (~5 seconds)
bash scripts/reproduce_all.sh           # full repro: ~7 GPU-h on T4
```

All paper figures and tables will be generated in `results/figures/`
and `results/*.csv`.

## Environment

- **Python**: 3.10 - 3.12 recommended (3.13+ works for everything except
  `pandas-datareader`-based Fama-French download; the cached CSV path
  bypasses this).
- **PyTorch**: 2.1+ with CUDA 12.1 (T4) or CUDA 11.8 (A100).
- **GPU**: T4 16GB or A100 40GB recommended. CPU-only works but is
  ~20× slower.
- **OS**: Tested on Ubuntu 22.04 (Google Colab) and Windows 11.

## Data sources

All data is publicly available; no API keys required for the headline
experiments.

| Source | What we use | Access |
|---|---|---|
| Yahoo Finance v8 | ETF daily OHLCV (80 tickers) | `yfinance`, cached locally |
| Kenneth French Data Library | FF 3 + 5 + Momentum daily factors | `pandas-datareader` |
| AQR Capital | Commodity Value/Momentum factors | Public CSV (research portal) |
| GDELT 2.0 / RSS | Financial news headlines | Public APIs |
| HuggingFace `ProsusAI/finbert` | FinBERT embeddings | Public model |

Full attribution and licensing in
[`data/croissant/dhrp-8universe.json`](data/croissant/dhrp-8universe.json)
(Croissant 1.1 metadata).

## Single-command reproduction

```bash
bash scripts/reproduce_all.sh
```

This script:
1. Verifies Python + CUDA
2. Installs dependencies if missing
3. Creates required result directories
4. Runs `notebooks/llm_dhrp_experiments.ipynb` end-to-end via
   `jupyter nbconvert --execute`
5. Validates Croissant 1.1 metadata
6. Verifies expected output files exist
7. Prints headline Sharpe + PSR pivots

## Per-cell reproduction (Jupyter)

Open `notebooks/llm_dhrp_experiments.ipynb` and run cells in order.
Cells are idempotent for a given seed.

| Cell | Phase |
|---|---|
| 1 | Setup: CUDA verify, deps, Drive mount (Colab) |
| 2 | Price data: download 80 tickers, cache |
| 3-4 | Headlines + FinBERT extraction |
| 5 | Optional Gemini sentiment (needs `GOOGLE_API_KEY`) |
| 6 | Temporal text features (90-day rolling windows) |
| 7 | FRED + macro features |
| 8-9 | Train DHRP + LLM-DHRP (single seed sanity) |
| 13-14 | Deep baselines + full backtest |
| 15-19 | Statistical tests + ablations |
| 23 | **Multi-seed expansion: 10 seeds × 8 universes** (paper headline) |
| 24 | Probe analysis (interpretability) |

## Random seeds

All neural methods use seeds `0..9` (10 seeds). PyTorch and NumPy seeds
are set at the start of each training function. Classical baselines
(EW, MV, MINVAR, HRP, RP, MAXDIV) are deterministic — they produce
identical output across runs and are reported without standard
deviation.

## Compute budget

| Step | GPU-h (T4) | GPU-h (A100) |
|---|---|---|
| FinBERT extraction (Cell 4) | 0.5 | 0.1 |
| Single-seed training (Cells 8-9, 13) | 0.5 | 0.1 |
| Multi-seed × 8 universes (Cell 23) | 6.0 | 1.5 |
| Statistical analysis (Cells 11, 14-19) | 0.2 CPU | — |
| Probe analysis (Cell 24) | 0.1 | 0.05 |
| **Total** | **~7.3 GPU-h** | **~1.8 GPU-h** |

Plus ~30 min CPU for figure generation, table compilation, and
Croissant validation.

## Verification checklist

After `reproduce_all.sh` completes, verify:

- [ ] `results/sharpe_pivot_multiseed_mean.csv`: DHRP Sharpe ≥ 0.95
      in Commodities, ≥ 1.20 in Crypto
- [ ] `results/psr_pivot_multiseed.csv`: DHRP PSR > 0.95 in Commodities
      and Crypto
- [ ] `results/Commodities_results.csv`: DHRP FF3 alpha t-stat > 2.0
- [ ] All 8 universes have non-empty multi-seed result CSVs
- [ ] `results/figures/cumulative_returns.png` renders cleanly

## Building the paper PDF

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Body fits in 6 pages (NeurIPS limit is 9). Output: `paper/main.pdf`.
