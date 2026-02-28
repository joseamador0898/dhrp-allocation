"""Smoke tests for all core modules."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

# --- Models ---
from src.models.dhrp_layer import DHRPLayer
from src.models.llm_dhrp_layer import LLMDHRPLayer
from src.models.baselines import equal_weight, hrp_allocation, risk_parity, max_diversification
from src.models.deep_baselines import MLPWithCovPolicy
from src.models.loss_functions import dhrp_loss

# --- Data ---
from src.data.feature_engineering import make_features, build_dataset
from src.data.fred_loader import load_fred_data, make_macro_features

# --- Evaluation ---
from src.evaluation.statistics import compute_sharpe, diebold_mariano_test
from src.evaluation.backtest import rolling_backtest
from src.evaluation.factor_analysis import factor_analysis

# --- Training ---
from src.training.trainer import train_dhrp, train_llm_dhrp, dhrp_weights

# Test DHRPLayer
model = DHRPLayer(n_assets=5, feature_dim=48, hidden_dim=64)
x = torch.randn(48)
S = torch.eye(5) * 0.04
w = model(x, S)
assert w.shape == (5,), f"Expected shape (5,), got {w.shape}"
assert abs(w.sum().item() - 1.0) < 1e-4
print(f"DHRPLayer: OK (params={sum(p.numel() for p in model.parameters())})")

# Test LLMDHRPLayer
llm_model = LLMDHRPLayer(n_assets=5, feature_dim=48, text_dim=768, use_text=True, use_macro=True, macro_dim=8)
w = llm_model(x, S, text_emb=torch.randn(768), macro_feat=torch.randn(8))
assert w.shape == (5,)
assert abs(w.sum().item() - 1.0) < 1e-4
print(f"LLMDHRPLayer: OK (params={sum(p.numel() for p in llm_model.parameters())})")

# Test MLPWithCovPolicy
mlp = MLPWithCovPolicy(feature_dim=48, n_assets=5)
w = mlp(x, S)
assert w.shape == (5,)
print(f"MLPWithCovPolicy: OK (params={sum(p.numel() for p in mlp.parameters())})")

# Test baselines
mu = np.array([0.1, 0.08, 0.05, 0.03, 0.02])
cov = np.eye(5) * 0.04
assert abs(equal_weight(mu, cov).sum() - 1.0) < 1e-6
assert abs(hrp_allocation(cov).sum() - 1.0) < 1e-6
assert abs(risk_parity(cov).sum() - 1.0) < 1e-6
print("Baselines: OK")

# Test Sharpe
r = np.random.randn(252) * 0.01 + 0.0003
sr = compute_sharpe(r)
print(f"Sharpe: OK ({sr:.3f})")

print("\nAll tests passed!")
