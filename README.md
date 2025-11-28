# Differentiable Hierarchical Risk Parity

This repo contains a compact public-data implementation of Differentiable Hierarchical Risk Parity (DHRP) and the LaTeX source of the associated paper.

The goal is simple. Start from classical hierarchical risk parity, keep its diversification logic, and turn it into a fully differentiable allocation rule that can be trained end to end with modern gradient methods.

## Structure

- `hrp_research_compact.ipynb` – single notebook that
  - downloads public ETF data for developed and emerging markets,
  - implements EW, MINVAR, MV, HRP, and DHRP,
  - trains the DHRP layer on rolling windows,
  - runs backtests for both universes,
  - computes statistics, factor regressions, and econometric diagnostics,
  - writes tables and figures to `results/`.
- `results/`
  - `hrp_paper_fixed.tex` – paper source: *Differentiable Hierarchical Risk Parity: Learning to Allocate with End-to-End Gradient Optimization*.
  - `DM_*.csv`, `EM_*.csv` – exported performance tables from recent runs.
  - `*.png` – summary plots, correlation heatmaps, return distributions, rolling Sharpe, cumulative wealth and drawdowns, risk–return scatter, and rolling volatility.

The notebook and paper are aligned. The definitions, loss functions, and hyperparameters in the code match what is described in the text, and the main numbers in the tables and figures come from the same backtest.

## Setup

You can run everything in a local virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

The notebook expects a working internet connection to

- download ETF prices via `yfinance`, and
- pull Fama–French daily factors via `pandas-datareader`.

## Running the experiments

1. Open `hrp_research_compact.ipynb` in Jupyter or VS Code.
2. Run cells from top to bottom.
3. The DHRP layer will train for developed markets first, then for emerging markets.
4. The backtest will compare EW, MINVAR, MV, HRP, and DHRP in both universes.
5. Tables and figures will appear under `results/` with a timestamp in the file names.

The main summary tables for Sharpe ratios, HAC t-statistics, bootstrap intervals, and Fama–French alphas are saved as CSV and printed in the notebook output. All plots used in the paper are also written to disk.

## Reproducing the paper

To keep the workflow simple:

- Use the notebook to regenerate
  - `DM_*.csv` and `EM_*.csv` tables, and
  - all `.png` figures.
- Compile `results/hrp_paper_fixed.tex` with your usual LaTeX toolchain.

The text, references, and figure captions in `hrp_paper_fixed.tex` are written to match the public-data notebook.

## NeurIPS-style repo layout

This layout follows a pattern common in NeurIPS submissions:

- one main notebook or script that reproduces the core results,
- a `results/` directory with tables and figures referenced in the paper,
- a clean `requirements.txt` to make the environment easy to rebuild,
- a short README that explains what the code does, how to run it, and how it connects to the paper.

If you want to extend the project, a natural next step is to split the notebook into a small Python package (`src/`) with

- a reusable DHRP layer,
- a backtest driver that can be called from the command line,
- a script that regenerates all tables and plots with one command.

For the current scope, keeping everything in one well-documented notebook keeps the repo easy to clone and run.
