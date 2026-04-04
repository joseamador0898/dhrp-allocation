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
        self.depth = depth
        self.is_em = is_em
        self.n_leaves = 2 ** depth
        self.n_internal = 2 ** depth - 1

        # Learnable soft leaf assignment: each asset has an affinity for each leaf.
        # Initialized with a warm start near the old modulo mapping so the tree
        # structure is preserved but assets can learn to redistribute.
        init_logits = torch.full((n_assets, self.n_leaves), -2.0)
        for i in range(n_assets):
            init_logits[i, i % self.n_leaves] = 2.0
        self.leaf_assign_logits = nn.Parameter(init_logits)

        # Precompute root-to-leaf paths for arbitrary depth
        self._paths = []
        for leaf in range(self.n_leaves):
            path = []
            node = 0
            bits = leaf
            for d in range(depth - 1, -1, -1):
                direction = (bits >> d) & 1
                path.append((node, direction))
                node = 2 * node + 1 + direction
            self._paths.append(path)

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

        # Higher init → sigmoid ≈ 0.82 → tree gets most of the weight
        self.vol_weight = nn.Parameter(torch.tensor(1.5 if not is_em else 0.5))

        # Learnable temperature for routing sharpness
        self.log_temp = nn.Parameter(torch.tensor(0.0 if not is_em else 0.7))

    def forward(self, x_t, Sigma_t):
        Sigma_norm = Sigma_t / (Sigma_t.abs().max() + 1e-6)
        node_feat = self.feat_norm(x_t + self.cov_proj(Sigma_norm.reshape(-1)))

        temp = torch.clamp(self.log_temp.exp(), 0.1, 5.0)
        probs = [F.softmax(g(node_feat) / temp, dim=-1) for g in self.gates]

        leaf_w = torch.stack(
            [
                torch.prod(torch.stack([probs[n][d] for n, d in p]))
                for p in self._paths
            ]
        )
        leaf_b = F.softmax(self.leaf_logits, dim=0) * leaf_w
        leaf_b = leaf_b / (leaf_b.sum() + 1e-8)

        # Soft assignment: each asset attends to all leaves with learned affinities
        assign = F.softmax(self.leaf_assign_logits, dim=-1)   # (n_assets, n_leaves)
        asset_b = (assign * leaf_b.unsqueeze(0)).sum(dim=-1)   # (n_assets,)
        asset_b = asset_b / (asset_b.sum() + 1e-8)

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
