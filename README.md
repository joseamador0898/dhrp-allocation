# DHRP: A Differentiable Hierarchical Risk Parity Architecture and Multi-Universe Portfolio Benchmark

NeurIPS 2026 Evaluations & Datasets track submission.

DHRP turns López de Prado's Hierarchical Risk Parity into a fully differentiable
portfolio allocation layer (temperature-scaled soft-gating tree, recovers
classical HRP as $\tau \to 0$). The DHRP-8 benchmark evaluates 11 allocators
across 8 asset universes with multi-seed Sharpe + Probabilistic Sharpe Ratio
reporting.

## Repo layout

```
paper/                 LaTeX source + final PDF (NeurIPS 2026 E&D format)
  main.tex             entry point
  sections/            8 body sections + appendix
  tables/              auto-generated from results/*.csv
  figures/             TikZ architecture diagram
  references.bib       52 verified entries
notebooks/
  llm_dhrp_experiments.ipynb    canonical experiment notebook (cells 1-24)
src/                   importable Python package
  data/                price + text + macro feature loaders, universe configs
  models/              dhrp_layer, llm_dhrp_layer, baselines, deep_baselines
  training/            multi-seed trainer
  evaluation/          backtest, statistics (PSR/SPA/MCS/DM/HAC), factor regs
  visualization/       plots
scripts/               7 user-facing scripts (see below)
results/               headline CSVs (multi-seed pivots) + figures
data/croissant/        Croissant 1.1 metadata for the DHRP-8 benchmark
```

## Reproducing the paper

```bash
pip install -r requirements.txt
bash scripts/reproduce_all.sh   # ~7 GPU-h on T4, ~2 GPU-h on A100
```

This runs `notebooks/llm_dhrp_experiments.ipynb` end-to-end, validates
Croissant metadata, and verifies the headline result CSVs exist. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for full per-seed-universe runtime
details and [SUBMISSION.md](SUBMISSION.md) for the day-by-day submission
checklist.

## Building the paper

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Body fits in 6 pages (NeurIPS limit is 9). Output: `paper/main.pdf`.

## Scripts

| Script | Purpose |
|---|---|
| `reproduce_all.sh` | Single-command end-to-end reproduction |
| `generate_paper_tables.py` | Build LaTeX tables from `results/*.csv` |
| `generate_paper_figures.py` | Build PDF figures from `results/*.csv` |
| `validate_paper.py` | Pre-submission integrity checks |
| `anonymize_check.py` | Double-blind de-anonymization scanner |
| `diagnose_llm_dhrp.py` | Optional: LLM-DHRP gate diagnostics (notebook cell 24) |
| `probe_analysis.py` | Optional: linear-probe interpretability (notebook cell 24) |

## Headline results (10 seeds, OOS 2020-07 to 2026-04)

| Universe | DHRP Sharpe | PSR | Best baseline |
|---|---|---|---|
| Commodities | $1.30 \pm 0.10$ | 0.999 | RP (1.11) |
| Crypto | $1.29 \pm 0.05$ | 0.991 | MV (1.11) |
| DM | $0.45 \pm 0.06$ | 0.856 | MV (0.33) |

DHRP beats classical HRP in 7 of 8 universes. LLM-augmented variant
(LLM-DHRP) is reported as an honest negative ablation.
