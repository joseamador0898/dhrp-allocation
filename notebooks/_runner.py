#!/usr/bin/env python3
"""Auto-generated notebook runner — upgraded pipeline with volume, extended FRED,
expanded headlines, Transformer/PPO baselines, BCa bootstrap, FDR, FF5+Mom.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Cell 1: SETUP ---
import warnings; warnings.filterwarnings("ignore")
import glob
import torch, numpy as np


def clean_stale_results():
    """Delete all stale results from previous runs."""
    patterns = [
        "results/*.csv", "results/*.png",
        "results/models/*.pt",
        "results/features/*.npz", "results/features/*.csv",
        "results/full/*.csv",
        "results/figures/*.png",
    ]
    total = 0
    for pat in patterns:
        for f in glob.glob(pat):
            os.remove(f)
            total += 1
    if total:
        print(f"Cleaned {total} stale result files.")


clean_stale_results()

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# --- Cell 2: LOAD PRICE + VOLUME DATA ---
from datetime import datetime, timedelta
from src.data.price_loader import load_universe, load_fama_french, load_etf_volume_data, UNIVERSES
from dotenv import load_dotenv
load_dotenv()

END = datetime.now().strftime('%Y-%m-%d')
START = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
print(f'Period: {START} to {END}\n')

print(f'=== DM Universe ({len(UNIVERSES["DM"])} ETFs) ===')
DM_prices = load_universe('DM', START, END)
print(f'\n=== EM Universe ({len(UNIVERSES["EM"])} ETFs) ===')
EM_prices = load_universe('EM', START, END)
print(f'\n=== Commodities Universe ({len(UNIVERSES["Commodities"])} ETFs) ===')
CMD_prices = load_universe('Commodities', START, END)

# Volume data
print('\n=== VOLUME DATA ===')
DM_volume = load_etf_volume_data(UNIVERSES['DM'], START, END)
EM_volume = load_etf_volume_data(UNIVERSES['EM'], START, END)
CMD_volume = load_etf_volume_data(UNIVERSES['Commodities'], START, END)

# Fama-French factors
print('\n=== Fama-French 5 + Momentum ===')
from src.evaluation.factor_analysis import load_ff5_factors
try:
    FF = load_ff5_factors(START, END)
except Exception:
    print('  FF5 failed, falling back to FF3...')
    FF = load_fama_french(START, END)

# --- Cell 3: LOAD HEADLINES (expanded) ---
from src.data.text_loader import load_all_headlines, load_yfinance_news
import pandas as pd

all_tickers = (
    list(UNIVERSES['DM'].values()) +
    list(UNIVERSES['EM'].values()) +
    list(UNIVERSES['Commodities'].values())
)
all_tickers = list(set(all_tickers))

print(f'\n=== HEADLINES ({len(all_tickers)} tickers) ===')
headlines_df = load_all_headlines(
    all_tickers, START, END, max_headlines=100,
    use_rss=True, use_phrasebank=True, use_fiqa=True,
)
print(f'Total unique headlines: {len(headlines_df)}')

# --- Cell 4: FINBERT EMBEDDINGS ---
from src.data.llm_features import get_finbert_embeddings

headline_to_emb = {}
if not headlines_df.empty:
    unique_headlines = headlines_df['headline'].unique().tolist()
    print(f'\nExtracting FinBERT embeddings for {len(unique_headlines)} unique headlines...')
    finbert_embs = get_finbert_embeddings(unique_headlines, batch_size=64, device=device)
    print(f'FinBERT embeddings shape: {finbert_embs.shape}')
    headline_to_emb = {h: finbert_embs[i] for i, h in enumerate(unique_headlines)}
else:
    print('No headlines available. Proceeding with price-only features.')

# --- Cell 5: QWEN3 STRUCTURED SENTIMENT (auto-selects 32B on A100, 8B on T4) ---
from src.data.llm_features import _select_qwen_model, get_qwen3_sentiment, sentiment_to_features

qwen_model = _select_qwen_model()
sentiment_by_ticker = {}
if qwen_model and headline_to_emb:
    print(f'\n=== STRUCTURED SENTIMENT ({qwen_model}) ===')
    tickers_with_headlines = headlines_df.groupby('ticker')['headline'].apply(list).to_dict()
    batch_inputs = []
    batch_tickers = []
    for ticker, hdls in tickers_with_headlines.items():
        if len(hdls) >= 3:  # only process tickers with enough headlines
            batch_inputs.append(hdls[:15])
            batch_tickers.append(ticker)

    if batch_inputs:
        sentiments = get_qwen3_sentiment(batch_inputs, device=device, model_name=qwen_model)
        for ticker, sent in zip(batch_tickers, sentiments):
            sentiment_by_ticker[ticker] = sentiment_to_features(sent)
        print(f'  Sentiment extracted for {len(sentiment_by_ticker)} tickers')
        if torch.cuda.is_available():
            print(f'  VRAM after Qwen3: {torch.cuda.memory_allocated()/1e9:.2f} GB')
else:
    if not qwen_model:
        print("\nNo GPU available for Qwen3 sentiment. Skipping.")
    else:
        print("\nNo headlines for Qwen3 sentiment. Skipping.")

# --- Cell 6: BUILD TEXT FEATURE TENSORS ---
from src.data.feature_engineering import build_dataset, DEFAULT_FDIM

def build_text_tensor_for_universe(prices, universe_tickers, headline_to_emb, headlines_df, volume=None):
    """Build (n_samples, n_assets, 768) FinBERT tensor for a universe."""
    X, S, R, H = build_dataset(prices, volume=volume, fdim=DEFAULT_FDIM)
    n_samp = X.shape[0]
    n_assets = prices.shape[1]
    ticker_list = list(universe_tickers.values())

    asset_embs = np.zeros((n_assets, 768), dtype=np.float32)
    for j, ticker in enumerate(ticker_list):
        ticker_headlines = headlines_df[headlines_df['ticker'] == ticker]['headline'].tolist()
        embs = [headline_to_emb[h] for h in ticker_headlines if h in headline_to_emb]
        if embs:
            asset_embs[j] = np.mean(embs, axis=0)

    text_tensor = np.tile(asset_embs, (n_samp, 1, 1))
    return text_tensor

text_dm, text_em, text_cmd = None, None, None

if headline_to_emb:
    print('\nBuilding text feature tensors...')
    text_dm = build_text_tensor_for_universe(DM_prices, UNIVERSES['DM'], headline_to_emb, headlines_df, DM_volume)
    text_em = build_text_tensor_for_universe(EM_prices, UNIVERSES['EM'], headline_to_emb, headlines_df, EM_volume)
    text_cmd = build_text_tensor_for_universe(CMD_prices, UNIVERSES['Commodities'], headline_to_emb, headlines_df, CMD_volume)
    print(f'  DM text: {text_dm.shape}')
    print(f'  EM text: {text_em.shape}')
    print(f'  CMD text: {text_cmd.shape}')

    os.makedirs('results/features', exist_ok=True)
    np.savez_compressed('results/features/text_dm.npz', finbert=text_dm)
    np.savez_compressed('results/features/text_em.npz', finbert=text_em)
    np.savez_compressed('results/features/text_cmd.npz', finbert=text_cmd)
    print('Saved to results/features/')
else:
    print('No text features available. Using price-only mode.')

# --- Cell 7: PER-UNIVERSE MACRO / SUPPLEMENTARY FEATURES ---
from src.data.fred_loader import (
    load_fred_data, make_macro_features,
    make_commodity_features, make_em_features,
    FRED_SERIES_COMMODITY, FRED_SERIES_EM,
)
from src.data.universe_config import UNIVERSE_DATA_CONFIG

print('\n=== PER-UNIVERSE DATA SOURCES ===')

# DM: full US FRED macro (VIX, yield curve, credit, CPI, etc.)
dm_macro = None
if UNIVERSE_DATA_CONFIG["DM"]["use_macro"]:
    print('\n--- DM: Loading US FRED macro features ---')
    fred_df = load_fred_data(START, END, extended=True)
    if not fred_df.empty:
        dm_macro = make_macro_features(fred_df)
        print(f'  DM macro: {dm_macro.shape} ({dm_macro.columns.tolist()[:8]}...)')
    else:
        print('  FRED not available.')

# GS Quant macro (DM only): SPX, VIX, NDX, RTY, MXEF, MXWO, MXEA, BCOMTR, DXY, EURUSD, USDJPY
try:
    from src.data.gsquant_loader import load_gs_data
    gs_df = load_gs_data(START, END)
    if not gs_df.empty:
        if dm_macro is not None:
            dm_macro = pd.concat([dm_macro, gs_df], axis=1, join='outer').ffill().bfill()
        else:
            dm_macro = gs_df
        print(f'  DM macro (FRED+GS): {dm_macro.shape}')
except Exception as e:
    print(f'  GS Quant: {e}')

# EM: NO US macro — text-only conditioning
print('\n--- EM: No US macro (text-only LLM-DHRP) ---')

# Commodities: NO US macro — text-only conditioning; SPGCI best-effort
print('\n--- Commodities: No US macro (text-only LLM-DHRP) ---')

if UNIVERSE_DATA_CONFIG["Commodities"]["spgci"]:
    try:
        from src.data.spgci_loader import load_spgci_data
        spgci_df = load_spgci_data(START, END)
        if not spgci_df.empty:
            print(f'  SPGCI features: {spgci_df.shape}')
    except Exception as e:
        print(f'  SPGCI: {e}')

# --- Cell 8: TRAIN ALL MODELS (DM Universe) ---
from src.training.trainer import train_dhrp, train_llm_dhrp
from src.models.deep_baselines import train_transformer_policy, train_ppo_agent

print('\n=== DEVELOPED MARKETS ===')
fdim = DEFAULT_FDIM

print('\n--- Training DHRP (price + volume) ---')
dhrp_dm = train_dhrp(DM_prices, device=device, is_em=False, volume=DM_volume, fdim=fdim)

llm_dhrp_dm = None
if text_dm is not None:
    print('\n--- Training LLM-DHRP DM (price + text + macro + volume) ---')
    llm_dhrp_dm = train_llm_dhrp(
        DM_prices,
        text_features={'finbert': text_dm},
        macro_features=dm_macro.values if dm_macro is not None else None,
        device=device, is_em=False, volume=DM_volume, fdim=fdim,
        use_text=True, use_macro=dm_macro is not None,
        fusion_type='cross_attention', depth=3,
        epochs=60, lr=3e-4,
    )

print('\n--- Training Transformer baseline ---')
transformer_dm = train_transformer_policy(DM_prices, device=device, is_em=False, volume=DM_volume, fdim=fdim)

print('\n--- Training PPO baseline ---')
ppo_dm = train_ppo_agent(DM_prices, device=device, is_em=False, volume=DM_volume, fdim=fdim, epochs=30)

os.makedirs('results/models', exist_ok=True)
torch.save(dhrp_dm.state_dict(), 'results/models/dhrp_dm.pt')
if llm_dhrp_dm is not None:
    torch.save(llm_dhrp_dm.state_dict(), 'results/models/llm_dhrp_dm.pt')
torch.save(transformer_dm.state_dict(), 'results/models/transformer_dm.pt')
torch.save(ppo_dm.state_dict(), 'results/models/ppo_dm.pt')
print('\nDM models saved.')

# --- Cell 9: TRAIN EM + COMMODITIES ---
print('\n=== EMERGING MARKETS ===')
print('\n--- Training DHRP (EM) ---')
dhrp_em = train_dhrp(EM_prices, device=device, is_em=True, volume=EM_volume, fdim=fdim)

llm_dhrp_em = None
if text_em is not None:
    print('\n--- Training LLM-DHRP EM (price + text, NO macro) ---')
    llm_dhrp_em = train_llm_dhrp(
        EM_prices, text_features={'finbert': text_em},
        macro_features=None,
        device=device, is_em=True, volume=EM_volume, fdim=fdim,
        use_text=True, use_macro=False,
        epochs=50, lr=1.5e-4,
    )

print('\n--- Training Transformer (EM) ---')
transformer_em = train_transformer_policy(EM_prices, device=device, is_em=True, volume=EM_volume, fdim=fdim)

print('\n=== COMMODITIES ===')
print('\n--- Training DHRP (Commodities) ---')
dhrp_cmd = train_dhrp(CMD_prices, device=device, is_em=False, volume=CMD_volume, fdim=fdim)

llm_dhrp_cmd = None
if text_cmd is not None:
    print('\n--- Training LLM-DHRP Commodities (price + text, NO macro) ---')
    llm_dhrp_cmd = train_llm_dhrp(
        CMD_prices, text_features={'finbert': text_cmd},
        macro_features=None,
        device=device, is_em=False, volume=CMD_volume, fdim=fdim,
        use_text=True, use_macro=False,
        epochs=60, lr=3e-4,
    )

print('\n--- Training Transformer (Commodities) ---')
transformer_cmd = train_transformer_policy(CMD_prices, device=device, is_em=False, volume=CMD_volume, fdim=fdim)

# Save all models
torch.save(dhrp_em.state_dict(), 'results/models/dhrp_em.pt')
torch.save(dhrp_cmd.state_dict(), 'results/models/dhrp_cmd.pt')
torch.save(transformer_em.state_dict(), 'results/models/transformer_em.pt')
torch.save(transformer_cmd.state_dict(), 'results/models/transformer_cmd.pt')
if llm_dhrp_em: torch.save(llm_dhrp_em.state_dict(), 'results/models/llm_dhrp_em.pt')
if llm_dhrp_cmd: torch.save(llm_dhrp_cmd.state_dict(), 'results/models/llm_dhrp_cmd.pt')
print('\nAll models saved.')

# --- Cell 10: ROLLING BACKTEST ---
from src.evaluation.backtest import rolling_backtest

METHODS = ['EW', 'MINVAR', 'MV', 'HRP', 'RP', 'MAXDIV', 'DHRP']
if llm_dhrp_dm is not None:
    METHODS.append('LLM_DHRP')

print('\n=== BACKTESTS ===')

dm_res = rolling_backtest(
    DM_prices, is_em=False, dhrp_model=dhrp_dm,
    llm_dhrp_model=llm_dhrp_dm, text_features={'finbert': text_dm} if text_dm is not None else None,
    macro_features=dm_macro, methods=METHODS, volume=DM_volume, purge_days=5,
)
print(f'DM: {len(dm_res)} observations')

em_res = rolling_backtest(
    EM_prices, is_em=True, dhrp_model=dhrp_em,
    llm_dhrp_model=llm_dhrp_em, text_features={'finbert': text_em} if text_em is not None else None,
    macro_features=None, methods=METHODS, volume=EM_volume, purge_days=5,
)
print(f'EM: {len(em_res)} observations')

cmd_res = rolling_backtest(
    CMD_prices, is_em=False, dhrp_model=dhrp_cmd,
    llm_dhrp_model=llm_dhrp_cmd, text_features={'finbert': text_cmd} if text_cmd is not None else None,
    macro_features=None, methods=METHODS, volume=CMD_volume, purge_days=5,
)
print(f'Commodities: {len(cmd_res)} observations')

# Transaction cost sensitivity
print('\n=== TRANSACTION COST SENSITIVITY ===')
for tc_bps in [10, 20, 50]:
    dm_tc = rolling_backtest(
        DM_prices, is_em=False, dhrp_model=dhrp_dm,
        llm_dhrp_model=llm_dhrp_dm, text_features={'finbert': text_dm} if text_dm is not None else None,
        macro_features=dm_macro, methods=['DHRP', 'LLM_DHRP', 'HRP'] if llm_dhrp_dm else ['DHRP', 'HRP'],
        volume=DM_volume, transaction_cost_bps=tc_bps, purge_days=5,
    )
    from src.evaluation.statistics import compute_stats as cs
    tc_stats = cs(dm_tc)
    print(f'  DM TC={tc_bps}bps: ' + ', '.join(f'{r["Method"]}={r["Sharpe"]:.3f}' for _, r in tc_stats.iterrows()))

# --- Cell 11: RESULTS & STATISTICAL TESTS ---
from src.evaluation.statistics import compute_stats, sharpe_difference_test, diebold_mariano_test, fdr_correct, subperiod_analysis
from src.evaluation.factor_analysis import factor_analysis

os.makedirs('results/full', exist_ok=True)

for label, res, prices, is_em, volume in [
    ('DM', dm_res, DM_prices, False, DM_volume),
    ('EM', em_res, EM_prices, True, EM_volume),
    ('Commodities', cmd_res, CMD_prices, False, CMD_volume),
]:
    print(f'\n{"="*60}')
    print(f'  {label} RESULTS ({prices.shape[1]} assets)')
    print(f'{"="*60}')

    stats = compute_stats(res, n_boot=1000)
    try:
        factors = factor_analysis(res, FF)
        table = stats.merge(factors, on='Method', how='left').round(3)
    except Exception:
        table = stats.round(3)
    print(table.to_string(index=False))

    # Statistical tests vs HRP with FDR correction
    print(f'\n--- Statistical Tests (vs HRP) ---')
    p_values = []
    test_methods = []
    for m in sorted(res['method'].unique()):
        if m == 'HRP':
            continue
        try:
            diff = sharpe_difference_test(res, m, 'HRP', n_boot=1000)
            dm_test = diebold_mariano_test(res, m, 'HRP')
            sig = ' ***' if diff['bootstrap_p'] < 0.01 else ' **' if diff['bootstrap_p'] < 0.05 else ' *' if diff['bootstrap_p'] < 0.10 else ''
            print(f"  {m:12s} vs HRP: diff={diff['sharpe_diff']:+.3f} "
                  f"[{diff['bootstrap_ci_lo']:.3f}, {diff['bootstrap_ci_hi']:.3f}] "
                  f"p={diff['bootstrap_p']:.3f}{sig} d={diff['cohens_d']:.3f} | "
                  f"DM={dm_test['DM_stat']:+.3f} p={dm_test['p_value']:.3f}")
            p_values.append(diff['bootstrap_p'])
            test_methods.append(m)
        except Exception as e:
            print(f'  {m:12s} vs HRP: failed ({e})')

    if p_values:
        fdr = fdr_correct(p_values)
        print(f'\n  FDR-adjusted p-values:')
        for name, raw, adj in zip(test_methods, fdr['raw'], fdr['adjusted']):
            sig = ' *' if adj < 0.10 else ''
            print(f"    {name:12s}: raw={raw:.3f} adj={adj:.3f}{sig}")

    # Sub-period analysis
    print(f'\n--- Sub-period Sharpe Ratios ---')
    sub = subperiod_analysis(res)
    if not sub.empty:
        pivot = sub.pivot(index='Method', columns='Period', values='Sharpe')
        print(pivot.round(3).to_string())

    table.to_csv(f'results/full/{label}_stats.csv', index=False)
    res.to_csv(f'results/full/{label}_backtest.csv', index=False)

# --- Cell 12: PAPER FIGURES ---
from src.visualization.plots import plot_cumulative, plot_sharpe_bars, get_series
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results/figures', exist_ok=True)
results_dict = {'DM': dm_res, 'EM': em_res, 'Commodities': cmd_res}
plot_cumulative(results_dict, output_dir='results/figures')
plot_sharpe_bars(results_dict, output_dir='results/figures')
print('\nCumulative returns and Sharpe bar figures saved.')

# LLM-DHRP vs DHRP delta
if 'LLM_DHRP' in dm_res['method'].unique():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (uname, res) in zip(axes, results_dict.items()):
        s_llm = get_series(res, 'LLM_DHRP')
        s_dhrp = get_series(res, 'DHRP')
        common = s_llm.index.intersection(s_dhrp.index)
        diff = s_llm.loc[common] - s_dhrp.loc[common]
        cum_diff = diff.cumsum() * 100
        ax.fill_between(common, cum_diff.values, 0,
                        where=cum_diff.values >= 0, alpha=0.3, color='green')
        ax.fill_between(common, cum_diff.values, 0,
                        where=cum_diff.values < 0, alpha=0.3, color='red')
        ax.plot(common, cum_diff.values, color='black', lw=1.5)
        ax.axhline(0, color='black', ls='--', lw=0.8)
        ax.set_title(f'{uname}: LLM-DHRP minus DHRP', fontweight='bold')
        ax.set_ylabel('Cumulative Excess Return (%)')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/llm_delta.png', dpi=300)
    plt.close()
    print('LLM delta figure saved.')

# --- Cell 13: GATING INTERPRETABILITY ---
if llm_dhrp_dm is not None:
    from src.data.feature_engineering import build_dataset, make_features

    X, S, R, H = build_dataset(DM_prices, volume=DM_volume, fdim=fdim)
    n_samples = min(50, X.shape[0])

    probs_with_text, probs_without_text = [], []
    routing_shifts = []

    for i in range(n_samples):
        x = torch.from_numpy(X[i]).to(device)
        s = torch.from_numpy(S[i]).to(device)
        te = torch.from_numpy(text_dm[i].mean(axis=0)).to(device) if text_dm is not None else torch.randn(768).to(device)

        p_with = llm_dhrp_dm.get_gating_probs(x, s, text_emb=te)
        p_without = llm_dhrp_dm.get_gating_probs(x, s, text_emb=None)
        probs_with_text.append([p.cpu().numpy() for p in p_with])
        probs_without_text.append([p.cpu().numpy() for p in p_without])

        shift = llm_dhrp_dm.get_routing_shift(x, s, text_emb=te)
        routing_shifts.append(shift)

    root_with = [p[0][0] for p in probs_with_text]
    root_without = [p[0][0] for p in probs_without_text]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(root_without, label='Price-only', alpha=0.7, color='steelblue')
    ax1.plot(root_with, label='Price + Text', alpha=0.7, color='crimson')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('P(left) at root node')
    ax1.set_title('Root Node Gating: Text Impact', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    total_shifts = [s['total'] for s in routing_shifts]
    ax2.bar(range(n_samples), total_shifts, color='darkorange', alpha=0.7)
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Total routing shift')
    ax2.set_title('Text Routing Shift Magnitude', fontweight='bold')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/gating_interpretability.png', dpi=300)
    plt.close()
    print('Gating interpretability figure saved.')

    gate_info = llm_dhrp_dm.get_text_gate_values(
        torch.from_numpy(X[0]).to(device),
        torch.from_numpy(S[0]).to(device),
        text_emb=torch.from_numpy(text_dm[0].mean(axis=0)).to(device) if text_dm is not None else None,
    )
    if gate_info:
        print(f'\nFusion gate: mean={gate_info["gate_mean"]:.3f} std={gate_info["gate_std"]:.3f}')
else:
    print('LLM-DHRP not trained. Skipping interpretability analysis.')

# --- Cell 14: FINAL SUMMARY ---
print('\n' + '='*70)
print('  EXPERIMENT COMPLETE')
print('='*70)

print(f'\nUniverses tested: DM ({DM_prices.shape[1]}), EM ({EM_prices.shape[1]}), CMD ({CMD_prices.shape[1]})')
print(f'Total assets: {DM_prices.shape[1] + EM_prices.shape[1] + CMD_prices.shape[1]}')
print(f'Feature dim: {fdim} (with volume features)')
print(f'Methods compared: {sorted(dm_res["method"].unique())}')
print(f'Headlines: {len(headlines_df)} unique')
print(f'DM macro features: {dm_macro.shape[1] if dm_macro is not None else 0}')
print(f'EM macro: None (text-only)')
print(f'CMD macro: None (text-only)')
print(f'Device used: {device}')

print('\nKey files saved:')
print('  results/full/{DM,EM,Commodities}_{stats,backtest}.csv')
print('  results/models/*.pt')
print('  results/features/*.npz')
print('  results/figures/*.png')

if torch.cuda.is_available():
    print(f'\nGPU memory peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
