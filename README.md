# DHRP: Differentiable Hierarchical Risk Parity

A differentiable extension of López de Prado's Hierarchical Risk Parity, plus
the **DHRP-8** multi-universe portfolio benchmark. Submitted to NeurIPS 2026
(Evaluations & Datasets track).

- DHRP recovers classical HRP as gate temperature $\tau \to 0$
- 11 allocators × 8 asset universes × 10 random seeds
- Headline: DHRP commodities Sharpe $1.30 \pm 0.10$ (PSR $0.999$)
- Honest negative ablation: FinBERT-augmented DHRP does not improve

## Quick start

```bash
git clone <this-repo>
cd dhrp-allocation
pip install -r requirements.txt
python tests/test_imports.py     # smoke test (~5 s, verifies environment)
```

You should see:

```
DHRPLayer: OK (params=32528)
LLMDHRPLayer: OK (params=157480)
MLPWithCovPolicy: OK (params=39189)
Baselines: OK
Sharpe: OK (...)
All tests passed!
```

## Reproducing the paper

```bash
bash scripts/reproduce_all.sh
```

Runs the canonical notebook end-to-end: ~7 GPU-h on T4, ~2 GPU-h on
A100. Outputs CSVs to `results/` and figures to `results/figures/`.

For a per-cell walkthrough, open
[`notebooks/llm_dhrp_experiments.ipynb`](notebooks/llm_dhrp_experiments.ipynb)
and run cells 1–24 in order. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
for cell-by-cell descriptions and compute budget.

## Verifying the headline numbers

After the notebook completes, the multi-seed pivot tables in
`results/sharpe_pivot_multiseed_mean.csv` and
`results/psr_pivot_multiseed.csv` should reproduce the paper's
Table 1:

| Universe | DHRP Sharpe | PSR | Best baseline |
|---|---|---|---|
| Commodities | $1.30 \pm 0.10$ | 0.999 | RP (1.11) |
| Crypto | $1.29 \pm 0.05$ | 0.991 | MV (1.11) |
| DM | $0.45 \pm 0.06$ | 0.856 | MV (0.33) |

DHRP beats classical HRP in 7 of 8 universes.

## Building the paper PDF

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
# → paper/main.pdf (15 pages: 6-page body + bibliography + checklist + appendix)
```

Requires a TeX distribution (MiKTeX on Windows, TeX Live on Linux/macOS).

## Repo layout

```
paper/                LaTeX source (NeurIPS 2026 E&D format)
notebooks/            Canonical experiment notebook (cells 1–24)
src/                  Importable Python package
  data/               price + text + macro feature loaders
  models/             dhrp_layer, llm_dhrp_layer, baselines
  training/           multi-seed trainer
  evaluation/         backtest, statistics, factor regressions
  visualization/      plots
scripts/              7 reviewer-facing scripts (see table below)
data/croissant/       Croissant 1.1 metadata for the DHRP-8 benchmark
results/              headline CSVs + figures (regenerated)
tests/                smoke tests
docs/                 supplementary theory notes
```

## Scripts

| Script | Purpose |
|---|---|
| `reproduce_all.sh` | Single-command end-to-end reproduction |
| `generate_paper_tables.py` | Build LaTeX tables from `results/*.csv` |
| `generate_paper_figures.py` | Build PDF figures from `results/*.csv` |
| `validate_paper.py` | Pre-submission integrity checks |
| `anonymize_check.py` | Double-blind de-anonymization scanner |
| `diagnose_llm_dhrp.py` | Optional LLM-DHRP gate diagnostics |
| `probe_analysis.py` | Optional linear-probe interpretability |

## Citation

Citation will be added on acceptance.

## License

[MIT](LICENSE).
