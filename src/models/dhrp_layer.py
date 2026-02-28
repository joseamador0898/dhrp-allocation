import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DHRPLayer(nn.Module):
    """Differentiable HRP with market-adaptive features."""

    def __init__(self, n_assets, feature_dim, hidden_dim=64, depth=3, is_em=False):
        super().__init__()
        self.n_assets = n_assets
        self.feature_dim = feature_dim
        self.is_em = is_em
        self.n_leaves = 2 ** depth
        self.n_internal = 2 ** depth - 1
        self.register_buffer(
            "asset_to_leaf", torch.arange(n_assets) % self.n_leaves
        )
        cov_h = hidden_dim if is_em else hidden_dim * 2
        gate_h = hidden_dim // 2 if is_em else hidden_dim
        self.feat_norm = nn.LayerNorm(feature_dim)
        self.cov_proj = nn.Sequential(
            nn.Linear(n_assets ** 2, cov_h), nn.Tanh(), nn.Linear(cov_h, feature_dim)
        )
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, gate_h),
                    nn.Tanh(),
                    nn.Linear(gate_h, 2),
                )
                for _ in range(self.n_internal)
            ]
        )
        self.leaf_logits = nn.Parameter(torch.zeros(self.n_leaves))
        self.vol_weight = nn.Parameter(torch.tensor(0.2 if is_em else 0.6))

    def forward(self, x_t, Sigma_t):
        Sigma_norm = Sigma_t / (Sigma_t.abs().max() + 1e-6)
        node_feat = self.feat_norm(x_t + self.cov_proj(Sigma_norm.reshape(-1)))
        temp = 3.0 if self.is_em else 1.5
        probs = [F.softmax(g(node_feat) / temp, dim=-1) for g in self.gates]
        # Hard-coded paths for depth-3 binary tree (8 leaves)
        paths = [
            [(0, 0), (1, 0), (3, 0)],
            [(0, 0), (1, 0), (3, 1)],
            [(0, 0), (1, 1), (4, 0)],
            [(0, 0), (1, 1), (4, 1)],
            [(0, 1), (2, 0), (5, 0)],
            [(0, 1), (2, 0), (5, 1)],
            [(0, 1), (2, 1), (6, 0)],
            [(0, 1), (2, 1), (6, 1)],
        ]
        leaf_w = torch.stack(
            [
                torch.prod(torch.stack([probs[n][d] for n, d in p]))
                for p in paths
            ]
        )
        leaf_b = F.softmax(self.leaf_logits, dim=0) * leaf_w
        leaf_b = leaf_b / (leaf_b.sum() + 1e-8)
        asset_b = torch.zeros(self.n_assets, device=x_t.device)
        for i in range(self.n_assets):
            asset_b[i] += leaf_b[int(self.asset_to_leaf[i].item())]
        vols = torch.sqrt(torch.clamp(torch.diag(Sigma_t), min=1e-8))
        inv_vol = (1.0 / vols) / ((1.0 / vols).sum() + 1e-8)
        alpha = torch.sigmoid(self.vol_weight)
        if self.is_em:
            cons_w = (
                0.5 * inv_vol
                + 0.5 * torch.ones(self.n_assets, device=x_t.device) / self.n_assets
            )
            final = alpha * asset_b + (1 - alpha) * cons_w
        else:
            final = alpha * asset_b + (1 - alpha) * inv_vol
        return torch.clamp(final, min=1e-6) / torch.clamp(final, min=1e-6).sum()
