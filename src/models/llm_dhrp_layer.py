"""LLM-enhanced DHRP layer with cross-modal fusion.

This module implements the key architectural contribution: LLM features
modulate the routing decisions in the soft gating tree. Financial text
like "Fed signals rate hike" changes which branch of the hierarchy
receives more allocation — language provides a regime routing signal
that aligns naturally with the tree structure.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextProjectionNetwork(nn.Module):
    """Projects LLM embeddings into the same space as price features."""

    def __init__(self, text_dim, target_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
            nn.LayerNorm(target_dim),
        )

    def forward(self, text_emb):
        return self.net(text_emb)


class CrossModalFusion(nn.Module):
    """Cross-attention fusion between price features and text features.

    Uses a residual gated mechanism: text provides an additive correction
    to price features, scaled by a learned gate initialized near zero
    so the model must earn the right to use text signals.
    """

    def __init__(self, feature_dim, n_heads=4, dropout=0.1, gate_bias_init=-2.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            feature_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.gate_proj = nn.Linear(feature_dim * 2, feature_dim)
        # Gate bias controls how much text is blended in at init.
        # -2.0 → sigmoid≈0.12 (DM/EM default), -0.5 → ≈0.38 (commodities).
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_bias_init)
        self.norm = nn.LayerNorm(feature_dim)
        self.text_dropout = nn.Dropout(0.3)

    def forward(self, price_feat, text_feat):
        """Fuse price and text features via residual gated cross-attention.

        Args:
            price_feat: (feature_dim,) price feature vector
            text_feat: (feature_dim,) projected text feature vector
        Returns:
            (feature_dim,) fused feature vector
        """
        p = price_feat.unsqueeze(0).unsqueeze(0) if price_feat.dim() == 1 else price_feat.unsqueeze(1)
        t = text_feat.unsqueeze(0).unsqueeze(0) if text_feat.dim() == 1 else text_feat.unsqueeze(1)

        attn_out, _ = self.cross_attn(p, t, t)
        attn_out = attn_out.squeeze(1).squeeze(0) if price_feat.dim() == 1 else attn_out.squeeze(1)
        attn_out = self.text_dropout(attn_out)

        # Residual gating: price_feat + gate * attn_correction
        gate_val = torch.sigmoid(self.gate_proj(torch.cat([price_feat, attn_out], dim=-1)))
        fused = price_feat + gate_val * (attn_out - price_feat)
        return self.norm(fused)


class LLMDHRPLayer(nn.Module):
    """LLM-enhanced Differentiable HRP.

    Architecture:
        price_features (48d) ──┐
                                ├── CrossModalFusion ──> DHRPTree ──> weights
        text_features  (48d) ──┘
                                        ↑
                             covariance_features

    The text features modulate which branches of the tree receive more
    allocation weight — providing a regime routing signal.
    """

    def __init__(
        self,
        n_assets,
        feature_dim=48,
        text_dim=768,
        hidden_dim=64,
        depth=3,
        is_em=False,
        use_text=True,
        use_macro=False,
        macro_dim=4,
        fusion_type="cross_attention",
        dropout=0.1,
        modality_dropout=0.2,
        gate_bias_init=-2.0,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.feature_dim = feature_dim
        self.text_dim = text_dim
        self.is_em = is_em
        self.use_text = use_text
        self.use_macro = use_macro
        self.modality_dropout = modality_dropout
        self.n_leaves = 2 ** depth
        self.n_internal = 2 ** depth - 1
        self.fusion_type = fusion_type

        # Learnable soft leaf assignment (same fix as DHRPLayer)
        init_logits = torch.full((n_assets, self.n_leaves), -2.0)
        for i in range(n_assets):
            init_logits[i, i % self.n_leaves] = 2.0
        self.leaf_assign_logits = nn.Parameter(init_logits)

        # Text projection: map LLM embeddings to feature_dim
        if use_text:
            self.text_proj = TextProjectionNetwork(
                text_dim, feature_dim, hidden_dim=hidden_dim * 2, dropout=dropout,
            )
            if fusion_type == "cross_attention":
                self.fusion = CrossModalFusion(
                    feature_dim, n_heads=4, dropout=dropout,
                    gate_bias_init=gate_bias_init,
                )
            elif fusion_type == "concat":
                self.fusion_proj = nn.Sequential(
                    nn.Linear(feature_dim * 2, feature_dim),
                    nn.LayerNorm(feature_dim),
                    nn.ReLU(),
                )
            # else: additive (no extra params)

        # Macro feature projection
        if use_macro:
            self.macro_proj = nn.Sequential(
                nn.Linear(macro_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, feature_dim),
            )
            self.macro_gate = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Sigmoid(),
            )

        # Covariance projection
        cov_h = hidden_dim if is_em else hidden_dim * 2
        self.feat_norm = nn.LayerNorm(feature_dim)
        self.cov_proj = nn.Sequential(
            nn.Linear(n_assets ** 2, cov_h),
            nn.Tanh(),
            nn.Linear(cov_h, feature_dim),
        )

        # Soft gating tree
        gate_h = hidden_dim // 2 if is_em else hidden_dim
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, gate_h),
                nn.Tanh(),
                nn.Linear(gate_h, 2),
            )
            for _ in range(self.n_internal)
        ])
        self.leaf_logits = nn.Parameter(torch.zeros(self.n_leaves))

        # Higher init for tree influence; learnable temperature
        self.vol_weight = nn.Parameter(torch.tensor(1.5 if not is_em else 0.5))
        self.log_temp = nn.Parameter(torch.tensor(0.0 if not is_em else 0.7))

        # Pre-compute tree paths
        self._paths = self._build_paths(depth)

    @staticmethod
    def _build_paths(depth):
        """Build root-to-leaf paths for a perfect binary tree."""
        n_leaves = 2 ** depth
        paths = []
        for leaf in range(n_leaves):
            path = []
            node = 0
            idx = leaf
            for d in range(depth):
                bit = (idx >> (depth - 1 - d)) & 1
                path.append((node, bit))
                node = 2 * node + 1 + bit
            paths.append(path)
        return paths

    def _fuse_features(self, price_feat, text_emb=None, macro_feat=None):
        """Fuse price, text, and macro features.

        During training, randomly drops text features (p=0.2) to prevent
        the model from becoming dependent on text and degrading price signals.
        """
        feat = price_feat

        if self.use_text and text_emb is not None:
            # Modality dropout: randomly skip text during training
            use_text_now = True
            if self.training and torch.rand(1).item() < self.modality_dropout:
                use_text_now = False

            if use_text_now:
                text_feat = self.text_proj(text_emb)
                if self.fusion_type == "cross_attention":
                    feat = self.fusion(feat, text_feat)
                elif self.fusion_type == "concat":
                    feat = self.fusion_proj(torch.cat([feat, text_feat], dim=-1))
                else:  # additive
                    feat = feat + 0.1 * text_feat  # scale down additive text

        if self.use_macro and macro_feat is not None:
            macro_projected = self.macro_proj(macro_feat)
            gate = self.macro_gate(torch.cat([feat, macro_projected], dim=-1))
            feat = feat + gate * macro_projected

        return feat

    def forward(self, x_t, Sigma_t, text_emb=None, macro_feat=None):
        """Forward pass.

        Args:
            x_t: (feature_dim,) price feature vector
            Sigma_t: (n_assets, n_assets) covariance matrix
            text_emb: (text_dim,) LLM embedding vector, or None
            macro_feat: (macro_dim,) macro feature vector, or None
        Returns:
            (n_assets,) portfolio weights summing to 1
        """
        # Fuse multimodal features
        fused_feat = self._fuse_features(x_t, text_emb, macro_feat)

        # Add covariance information
        Sigma_norm = Sigma_t / (Sigma_t.abs().max() + 1e-6)
        node_feat = self.feat_norm(fused_feat + self.cov_proj(Sigma_norm.reshape(-1)))

        # Soft gating tree with learnable temperature
        temp = torch.clamp(self.log_temp.exp(), 0.1, 5.0)
        probs = [F.softmax(g(node_feat) / temp, dim=-1) for g in self.gates]

        leaf_w = torch.stack([
            torch.prod(torch.stack([probs[n][d] for n, d in p]))
            for p in self._paths
        ])

        leaf_b = F.softmax(self.leaf_logits, dim=0) * leaf_w
        leaf_b = leaf_b / (leaf_b.sum() + 1e-8)

        # Soft assignment: each asset attends to all leaves
        assign = F.softmax(self.leaf_assign_logits, dim=-1)   # (n_assets, n_leaves)
        asset_b = (assign * leaf_b.unsqueeze(0)).sum(dim=-1)   # (n_assets,)
        asset_b = asset_b / (asset_b.sum() + 1e-8)

        # Inverse-volatility blending
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

    def get_gating_probs(self, x_t, Sigma_t, text_emb=None, macro_feat=None):
        """Return gating probabilities for interpretability analysis."""
        fused_feat = self._fuse_features(x_t, text_emb, macro_feat)
        Sigma_norm = Sigma_t / (Sigma_t.abs().max() + 1e-6)
        node_feat = self.feat_norm(fused_feat + self.cov_proj(Sigma_norm.reshape(-1)))
        temp = torch.clamp(self.log_temp.exp(), 0.1, 5.0)
        return [F.softmax(g(node_feat) / temp, dim=-1).detach() for g in self.gates]

    def get_routing_shift(self, x_t, Sigma_t, text_emb=None, macro_feat=None):
        """Quantify text impact on each node's routing decision.

        Compares gating probs with and without text features.
        Returns dict with per-node routing shift magnitudes.
        """
        probs_with = self.get_gating_probs(x_t, Sigma_t, text_emb, macro_feat)
        probs_without = self.get_gating_probs(x_t, Sigma_t, text_emb=None, macro_feat=macro_feat)

        shifts = {}
        for i, (pw, pwo) in enumerate(zip(probs_with, probs_without)):
            delta = (pw - pwo).abs().sum().item()
            shifts[f"node_{i}"] = delta
        shifts["total"] = sum(shifts.values())
        shifts["root"] = (probs_with[0] - probs_without[0])[0].item()  # P(left) shift at root
        return shifts

    def get_text_gate_values(self, x_t, Sigma_t, text_emb=None, macro_feat=None):
        """Return the fusion gate values (how much the model attends to text vs price).

        Only meaningful when fusion_type='cross_attention'.
        """
        if not self.use_text or text_emb is None:
            return None

        text_feat = self.text_proj(text_emb)
        if self.fusion_type == "cross_attention":
            # Replicate the gated fusion to extract gate values
            price_feat = x_t
            p = price_feat.unsqueeze(0).unsqueeze(0) if price_feat.dim() == 1 else price_feat.unsqueeze(1)
            t = text_feat.unsqueeze(0).unsqueeze(0) if text_feat.dim() == 1 else text_feat.unsqueeze(1)
            attn_out, attn_weights = self.fusion.cross_attn(p, t, t)
            attn_out = attn_out.squeeze(1).squeeze(0) if price_feat.dim() == 1 else attn_out.squeeze(1)
            gate_val = self.fusion.gate(torch.cat([price_feat, attn_out], dim=-1))
            return {
                "gate_mean": gate_val.mean().item(),
                "gate_std": gate_val.std().item(),
                "attn_weights": attn_weights.detach() if attn_weights is not None else None,
            }
        return None
