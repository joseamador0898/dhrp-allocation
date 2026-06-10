#!/usr/bin/env bash
# =====================================================================
# Single-command reproduction of all DHRP-8 paper results
#
# Usage:  bash scripts/reproduce_all.sh
#
# Total compute: ~7 GPU-hours on T4, ~2 GPU-hours on A100.
# =====================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

SMOKE_ONLY=0
if [[ "${1:-}" == "--smoke-only" ]]; then
    SMOKE_ONLY=1
elif [[ "${1:-}" != "" ]]; then
    echo "Usage: bash scripts/reproduce_all.sh [--smoke-only]"
    exit 2
fi

if command -v python > /dev/null 2>&1 && python -c "import sys; raise SystemExit(sys.version_info < (3, 10))"; then
    PYTHON_BIN=(python)
elif command -v python.exe > /dev/null 2>&1 && python.exe -c "import sys; raise SystemExit(sys.version_info < (3, 10))"; then
    PYTHON_BIN=(python.exe)
elif command -v py.exe > /dev/null 2>&1 && py.exe -3 -c "import sys; raise SystemExit(sys.version_info < (3, 10))"; then
    PYTHON_BIN=(py.exe -3)
elif command -v py > /dev/null 2>&1 && py -3 -c "import sys; raise SystemExit(sys.version_info < (3, 10))"; then
    PYTHON_BIN=(py -3)
elif command -v python3 > /dev/null 2>&1 && python3 -c "import sys; raise SystemExit(sys.version_info < (3, 10))"; then
    PYTHON_BIN=(python3)
else
    echo "Python 3.10+ not found on PATH"
    exit 1
fi

# 1. Verify Python before importing optional dependencies.
"${PYTHON_BIN[@]}" -c "
import sys
assert sys.version_info >= (3, 10), 'Python 3.10+ required'
print(f'Python: {sys.version.split()[0]}')
"

# 2. Install dependencies before importing torch or src modules.
if ! "${PYTHON_BIN[@]}" - <<'PY'
import importlib.util
required = ["torch", "sentence_transformers", "pydantic", "cvxpy", "pandas_datareader"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("Missing:", ", ".join(missing))
    raise SystemExit(1)
PY
then
    echo '>>> Installing dependencies from requirements.txt...'
    "${PYTHON_BIN[@]}" -m pip install -q -r requirements.txt
else
    echo '>>> Dependencies already installed.'
fi

"${PYTHON_BIN[@]}" -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# 3. Smoke test: verify imports, forward passes, and mini-panel integration.
echo '>>> Running smoke test...'
"${PYTHON_BIN[@]}" -u - <<'PY'
import os
import sys
import torch
import numpy as np
import pandas as pd

from src.models import DHRPLayer, LLMDHRPLayer, MLPWithCovPolicy, equal_weight, hrp_allocation, risk_parity
from src.evaluation import compute_sharpe, rolling_backtest, compute_stats
from src.data import build_dataset
from src.training import dhrp_weights

m = DHRPLayer(n_assets=5, feature_dim=48, hidden_dim=64)
w = m(torch.randn(48), torch.eye(5) * 0.04)
assert w.shape == (5,) and abs(w.sum().item() - 1.0) < 1e-4
print(f'  DHRPLayer OK ({sum(p.numel() for p in m.parameters())} params)')
m2 = LLMDHRPLayer(n_assets=5, feature_dim=48, text_dim=1536, use_text=True, use_macro=True, macro_dim=8)
w = m2(torch.randn(48), torch.eye(5) * 0.04, text_emb=torch.randn(1536), macro_feat=torch.randn(8))
assert w.shape == (5,)
print(f'  LLMDHRPLayer OK ({sum(p.numel() for p in m2.parameters())} params)')
mu, cov = np.array([0.1, 0.08, 0.05, 0.03, 0.02]), np.eye(5) * 0.04
assert all(abs(fn(mu, cov).sum() - 1.0) < 1e-6 if fn.__name__ == 'equal_weight' else abs(fn(cov).sum() - 1.0) < 1e-6 for fn in (equal_weight,))
assert abs(hrp_allocation(cov).sum() - 1.0) < 1e-6 and abs(risk_parity(cov).sum() - 1.0) < 1e-6
print('  Baselines OK')
print(f'  Sharpe OK ({compute_sharpe(np.random.randn(252) * 0.01 + 3e-4):.3f})')
print('  Lightweight smoke test OK', flush=True)
os._exit(0)

dates = pd.bdate_range('2020-01-01', periods=420)
rets = np.random.default_rng(0).normal(0.0002, 0.01, size=(len(dates), 5))
prices = pd.DataFrame(
    100 * np.exp(np.cumsum(rets, axis=0)),
    index=dates,
    columns=[f'A{i}' for i in range(5)],
)

X, S, R, H = build_dataset(prices, window=63, step=21, fdim=64)
assert X.shape[0] > 0, 'build_dataset returned no samples'
assert S.shape[1:] == (5, 5), S.shape
assert abs(hrp_allocation(S[0]).sum() - 1.0) < 1e-6

model = DHRPLayer(n_assets=5, feature_dim=X.shape[1], hidden_dim=16)
llm = LLMDHRPLayer(n_assets=5, feature_dim=X.shape[1], text_dim=1536, hidden_dim=16, use_text=True)
train_rets = prices.pct_change(fill_method=None).dropna().iloc[-63:]
w = dhrp_weights(model, train_rets)
assert w is not None and w.shape == (5,) and abs(w.sum() - 1) < 1e-5

text = {d: np.ones((5, 768), dtype=np.float32) for d in dates[80::21]}
res, diag = rolling_backtest(
    prices,
    dhrp_model=model,
    llm_dhrp_model=llm,
    text_features={'finbert': text},
    methods=['EW', 'HRP', 'DHRP', 'LLM_DHRP'],
    train_days=63,
    test_days=21,
    step_days=21,
    purge_days=2,
    strict_methods=['DHRP', 'LLM_DHRP'],
    return_diagnostics=True,
)
assert not res.empty, 'rolling_backtest returned empty results'
assert {'EW', 'HRP', 'DHRP', 'LLM_DHRP'}.issubset(set(res['method'])), res
assert diag['fallback_counts'] == {}, diag
assert diag['failure_counts'] == {}, diag
stats = compute_stats(res)
assert {'EW', 'HRP', 'DHRP', 'LLM_DHRP'}.issubset(set(stats['Method'])), stats
print('  CPU mini-panel smoke test OK')
print('  All imports + forward passes OK', flush=True)
os._exit(0)
PY

if [[ "$SMOKE_ONLY" == "1" ]]; then
    echo '>>> Smoke-only run complete.'
    exit 0
fi

# 4. Make sure result directories exist
mkdir -p results/figures results/models results/features results/full data/croissant

# 4. Convert notebook to executable script and run end-to-end
echo '>>> Running notebook end-to-end (jupyter nbconvert + execute)...'
"${PYTHON_BIN[@]}" -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=43200 \
    notebooks/llm_dhrp_experiments.ipynb

# 5. Validate Croissant metadata
echo '>>> Validating Croissant 1.1 metadata...'
"${PYTHON_BIN[@]}" -c "
import json
with open('data/croissant/dhrp-8universe.json') as f:
    meta = json.load(f)
assert meta['conformsTo'] == 'http://mlcommons.org/croissant/1.1'
assert 'rai:dataCollection' in meta
assert 'rai:dataBiases' in meta
assert len(meta['distribution']) >= 3
assert len(meta['recordSet']) >= 3
print('Croissant 1.1 metadata: OK')
"

# 6. Verify expected output files
echo '>>> Verifying expected output files...'
for f in \
    results/sharpe_pivot_multiseed_mean.csv \
    results/sharpe_pivot_multiseed_std.csv \
    results/psr_pivot_multiseed.csv \
    results/all_universes_multiseed_summary.csv \
    results/Commodities_results.csv \
    results/figures/cumulative_returns.png ; do
    if [ ! -f "$f" ]; then
        echo "  MISSING: $f"
        exit 1
    fi
done
echo '  All expected output files present.'

# 7. Summary
echo ''
echo '===================================================================='
echo '  REPRODUCTION COMPLETE'
echo '===================================================================='
echo "  Headline numbers (results/sharpe_pivot_multiseed_mean.csv):"
"${PYTHON_BIN[@]}" -c "
import pandas as pd
df = pd.read_csv('results/sharpe_pivot_multiseed_mean.csv', index_col=0)
print(df.round(3).to_string())
"
echo ''
echo '  Probabilistic Sharpe Ratios (results/psr_pivot_multiseed.csv):'
"${PYTHON_BIN[@]}" -c "
import pandas as pd
df = pd.read_csv('results/psr_pivot_multiseed.csv', index_col=0)
print(df.round(3).to_string())
"
echo ''
echo '  Paper artifacts:'
echo '    LaTeX source: paper/main.tex'
echo '    Figures: results/figures/'
echo '    Croissant: data/croissant/dhrp-8universe.json'
echo '===================================================================='
