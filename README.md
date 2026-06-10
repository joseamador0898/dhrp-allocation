# DHRP: Differentiable Hierarchical Risk Parity

A differentiable, HRP-inspired portfolio allocator plus the **DHRP-8** multi-universe portfolio benchmark. ICAIF 2026 primary submission package, with an AAAI-style backup draft retained for later general-AI submission.

- DHRP is an HRP-inspired differentiable relaxation; we do not claim
  the hard-routing limit reproduces full classical HRP.
- 8 headline methods across 8 asset universes; 12 allocators are
  implemented, with deep baselines reported in supplementary/appendix
  experiments where evaluated.
- Headline: DHRP commodities Sharpe $1.30 \pm 0.10$, PSR $0.999$
  (positive sample Sharpe; the FF3 alpha is $6.3\%$, $t=1.58$, not
  significant at the 5\% level).
- Honest negative ablation: a global FinBERT text summary does not
  improve DHRP under multi-seed reporting on the universes where text
  features are available (DM, EM, commodities).

## Quick start

```bash
git clone <this-repo>
cd dhrp-allocation
pip install -r requirements.txt
bash scripts/reproduce_all.sh     # smoke test + full run
bash scripts/reproduce_all.sh --smoke-only
```

The smoke path verifies imports, DHRP and LLM-DHRP forward passes, dataset construction, a CPU mini-panel DHRP backtest, and zero fallback weight computations for strict methods. The full run takes about 7 GPU-hours on a T4 or 2 GPU-hours on an A100, end-to-end.

## Repo layout

```
paper/                  LaTeX source
  main_icaif.tex        primary ACM/ICAIF anonymous submission source
  main_icaif.pdf        compiled primary ICAIF PDF
  main_aaai_submission.tex  AAAI-style backup draft (uses aaai2026 proxy style until 2027 kit releases)
  main_aaai_submission.pdf  compiled anonymous AAAI-style backup PDF
  references.bib        52 entries
  neurips_2026.sty      NeurIPS 2026 style file
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
  validate_claims.py    stale-claim and identity-pattern gate
  anonymize_check.py    double-blind de-anonymization scan
data/
  croissant/dhrp-8universe.json    Croissant 1.1 metadata
README.md, LICENSE, requirements.txt, requirements-frozen.txt, .gitignore
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
| 24 | Artifact validation and stale-claim checks |

### Compute budget

| Step | GPU-h on T4 | GPU-h on A100 |
|---|---|---|
| FinBERT extraction | 0.5 | 0.1 |
| Single-seed training | 0.5 | 0.1 |
| Multi-seed across 8 universes | 6.0 | 1.5 |
| Statistical analysis | 0.2 CPU | n/a |
| **Total** | **about 7.3** | **about 1.8** |

Plus about 30 minutes of CPU time for figures, table generation, and Croissant validation.

## Verifying the headline numbers

After the notebook completes, the multi-seed pivot tables in `results/sharpe_pivot_multiseed_mean.csv` and `results/psr_pivot_multiseed.csv` should match Table 1 in the paper:

| Universe | DHRP Sharpe | PSR | Best non-DHRP baseline |
|---|---|---|---|
| Commodities | $1.30 \pm 0.10$ | 0.999 | RP (1.11) |
| Crypto | $1.29 \pm 0.05$ | 0.991 | MV (1.11) |
| DM | $0.45 \pm 0.06$ | 0.856 | MV (0.33) |

DHRP beats classical HRP in 6 of 8 universes; the two losses (sectors
and factors) are within 0.03 Sharpe of HRP. PSR > 0.95 means the
out-of-sample Sharpe is significantly positive; it is not a paired test
against any specific baseline.

Verification checklist:

- [ ] DHRP Sharpe at least 0.95 in Commodities, at least 1.20 in Crypto
- [ ] DHRP PSR above 0.95 in both Commodities and Crypto
- [ ] DHRP FF3 alpha in Commodities is reported as about 6.3%, t≈1.58, not significant at 5%
- [ ] All 8 universes have non-empty multi-seed result CSVs
- [ ] Pairwise DHRP-vs-baseline tests are reported separately from PSR

## Building the paper PDF

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Requires a TeX distribution (MiKTeX on Windows, TeX Live on Linux or macOS). Primary output: `paper/main_icaif.pdf`. Build with `pdflatex main_icaif`, `bibtex main_icaif`, then two more `pdflatex main_icaif` runs. The backup AAAI-style draft is `paper/main_aaai_submission.pdf`.

## Environment

- **Python**: 3.10 to 3.12 recommended.
- **PyTorch**: 2.1+ with CUDA 12.1 (T4) or CUDA 11.8 (A100).
- **GPU**: T4 16 GB or A100 40 GB recommended; CPU-only works but is roughly 20x slower.
- **OS**: Tested on Ubuntu 22.04 (Google Colab) and Windows 11.

`requirements.txt` gives minimum compatible ranges. `requirements-frozen.txt`
captures the local environment used for artifact validation.

All raw market data is fetched from public upstream sources by the scripts; no raw proprietary archives are redistributed. Sources: Yahoo Finance ETF prices via `yfinance`, Fama-French factors via `pandas-datareader`, news headlines via GDELT 2.0 / RSS, and FinBERT embeddings derived from the `ProsusAI/finbert` HuggingFace model. Full attribution and licensing in `data/croissant/dhrp-8universe.json`.

## Citation

Will be added on acceptance.

## License

[MIT](LICENSE).


## AAAI 2027 submission note

The primary target is AAAI 2027 because its OpenReview venue currently lists an abstract-registration deadline of 2026-07-21 and a full-paper deadline of 2026-07-28. The current LaTeX uses AAAI 2026 style files as a proxy; before actual submission, replace them with the official AAAI 2027 author kit and recompile. Keep the main-paper claims conservative: PSR is evidence of positive Sharpe, not paired outperformance, and DHRP is an HRP-inspired differentiable relaxation rather than an exact differentiable implementation of classical HRP.
