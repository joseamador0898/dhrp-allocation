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

# 1. Verify Python + CUDA
python -c "
import sys, torch
assert sys.version_info >= (3, 10), 'Python 3.10+ required'
print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# 2. Install pinned dependencies (skip if already installed)
if ! pip show sentence-transformers > /dev/null 2>&1; then
    echo '>>> Installing dependencies from requirements-frozen.txt...'
    pip install -q -r requirements-frozen.txt
else
    echo '>>> Dependencies already installed.'
fi

# 3. Make sure result directories exist
mkdir -p results/figures results/models results/features results/full data/croissant

# 4. Convert notebook to executable script and run end-to-end
echo '>>> Running notebook end-to-end (jupyter nbconvert + execute)...'
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=43200 \
    notebooks/llm_dhrp_experiments.ipynb

# 5. Validate Croissant metadata
echo '>>> Validating Croissant 1.1 metadata...'
python -c "
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
python -c "
import pandas as pd
df = pd.read_csv('results/sharpe_pivot_multiseed_mean.csv', index_col=0)
print(df.round(3).to_string())
"
echo ''
echo '  Probabilistic Sharpe Ratios (results/psr_pivot_multiseed.csv):'
python -c "
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
