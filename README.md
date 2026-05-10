# DHRP: Differentiable Hierarchical Risk Parity

A differentiable extension of López de Prado's Hierarchical Risk Parity, plus the **DHRP-8** multi-universe portfolio benchmark. NeurIPS 2026 Evaluations & Datasets track submission.

- DHRP recovers classical HRP as gate temperature $\tau \to 0$
- 11 allocators, 8 asset universes, 10 random seeds
- Headline: DHRP commodities Sharpe $1.30 \pm 0.10$ (PSR $0.999$)
- Honest negative ablation: FinBERT-augmented DHRP does not improve

## Quick start

```bash
git clone <this-repo>
cd dhrp-allocation
pip install -r requirements.txt
bash scripts/reproduce_all.sh     # smoke test + full run
```

The first step of `reproduce_all.sh` is a 5-second import smoke test that verifies the environment. The full run takes about 7 GPU-hours on a T4 or 2 GPU-hours on an A100, end-to-end.

## Repo layout (20 files)

```
paper/                  LaTeX source (1086-line single-file main.tex)
  main.tex              all sections + tables + figure + appendix + checklist
  references.bib        52 entries
  neurips_2026.sty      NeurIPS 2026 style file
  main.pdf              built paper (15 pages)
notebooks/
  llm_dhrp_experiments.ipynb    canonical experiment notebook
src/                    flat Python package (no submodules)
  data.py               price + text + macro loaders, universe configs
  models.py             DHRPLayer, LLMDHRPLayer, baselines, deep baselines
  training.py           multi-seed trainer
  evaluation.py         backtest + statistics (PSR, SPA, MCS) + factor regs
  visualization.py      plots
scripts/
  reproduce_all.sh      single-command repro driver
  validate_paper.py     pre-submission integrity scan
  anonymize_check.py    double-blind de-anonymization scan
  diagnose_llm_dhrp.py  optional LLM-DHRP diagnostics (notebook cell 24)
  probe_analysis.py     optional interpretability probes (notebook cell 24)
data/
  croissant/dhrp-8universe.json    Croissant 1.1 metadata
  headlines/all_headlines.csv      cached news headlines (sample)
README.md, LICENSE, requirements.txt, .gitignore
```

## Reproducing the paper

```bash
bash scripts/reproduce_all.sh
```

The script runs `notebooks/llm_dhrp_experiments.ipynb` end-to-end, validates the Croissant metadata, and verifies the headline result CSVs are present. For a per-cell walkthrough, open the notebook directly.

| Cell | Phase |
|---|---|
| 1 | Setup: CUDA verify, dependencies |
| 2 | Price data: 80 ETFs from Yahoo Finance, cached |
| 3, 4 | Headlines + FinBERT extraction |
| 5 | Optional Gemini sentiment (needs `GOOGLE_API_KEY`) |
| 6 | Temporal text features (90-day rolling windows) |
| 7 | FRED + macro features |
| 8, 9 | Train DHRP and LLM-DHRP (single-seed sanity) |
| 13, 14 | Deep baselines + full backtest |
| 15 to 19 | Statistical tests + ablations |
| 23 | Multi-seed expansion: 10 seeds across 8 universes (paper headline) |
| 24 | Probe analysis (interpretability) |

### Compute budget

| Step | GPU-h on T4 | GPU-h on A100 |
|---|---|---|
| FinBERT extraction | 0.5 | 0.1 |
| Single-seed training | 0.5 | 0.1 |
| Multi-seed across 8 universes | 6.0 | 1.5 |
| Statistical analysis | 0.2 CPU | n/a |
| Probe analysis | 0.1 | 0.05 |
| **Total** | **about 7.3** | **about 1.8** |

Plus about 30 minutes of CPU time for figures, table generation, and Croissant validation.

## Verifying the headline numbers

After the notebook completes, the multi-seed pivot tables in `results/sharpe_pivot_multiseed_mean.csv` and `results/psr_pivot_multiseed.csv` should match Table 1 in the paper:

| Universe | DHRP Sharpe | PSR | Best baseline |
|---|---|---|---|
| Commodities | $1.30 \pm 0.10$ | 0.999 | RP (1.11) |
| Crypto | $1.29 \pm 0.05$ | 0.991 | MV (1.11) |
| DM | $0.45 \pm 0.06$ | 0.856 | MV (0.33) |

DHRP beats classical HRP in 7 of 8 universes.

Verification checklist:

- [ ] DHRP Sharpe at least 0.95 in Commodities, at least 1.20 in Crypto
- [ ] DHRP PSR above 0.95 in both Commodities and Crypto
- [ ] DHRP FF3 alpha t-statistic above 2.0 in Commodities
- [ ] All 8 universes have non-empty multi-seed result CSVs

## Building the paper PDF

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Requires a TeX distribution (MiKTeX on Windows, TeX Live on Linux or macOS). Output: `paper/main.pdf` (15 pages, body fits in 6 of the 9 allowed).

## Environment

- **Python**: 3.10 to 3.12 recommended.
- **PyTorch**: 2.1+ with CUDA 12.1 (T4) or CUDA 11.8 (A100).
- **GPU**: T4 16 GB or A100 40 GB recommended; CPU-only works but is roughly 20x slower.
- **OS**: Tested on Ubuntu 22.04 (Google Colab) and Windows 11.

All data is publicly available; no API keys required for the headline experiments. Sources: Yahoo Finance ETF prices via `yfinance`, Fama-French factors via `pandas-datareader`, news headlines via GDELT 2.0 / RSS, FinBERT embeddings via the `ProsusAI/finbert` HuggingFace model. Full attribution and licensing in `data/croissant/dhrp-8universe.json`.

## Citation

Will be added on acceptance.

## License

[MIT](LICENSE).
