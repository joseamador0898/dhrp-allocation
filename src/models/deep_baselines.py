"""Deep learning baselines for portfolio allocation ablation studies.

- MLPWithCovPolicy: Flat MLP (ablation A5 — does tree structure help?)
- TransformerPortfolioPolicy: Self-attention over assets (arXiv 2206.03246 style)
- PPOPortfolioAgent: Proximal Policy Optimization for portfolio allocation (SAPPO/ICML 2025 style)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPWithCovPolicy(nn.Module):
    """MLP that takes both features and covariance as input.

    Fair comparison with DHRP: same inputs (features + covariance), flat architecture.
    Used in ablation A5 to test whether tree structure helps exploit LLM features.
    """

    def __init__(self, feature_dim, n_assets, hidden=128, dropout=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.feature_dim = feature_dim
        cov_dim = n_assets * n_assets
        self.feat_norm = nn.LayerNorm(feature_dim)
        self.cov_proj = nn.Sequential(
            nn.Linear(cov_dim, hidden), nn.Tanh(), nn.Linear(hidden, feature_dim),
        )
        combined_dim = feature_dim * 2
        self.net = nn.Sequential(
            nn.Linear(combined_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_assets), nn.Softmax(dim=-1),
        )

    def forward(self, x_t, Sigma_t):
        """Forward pass matching DHRPLayer signature."""
        Sigma_norm = Sigma_t / (Sigma_t.abs().max() + 1e-6)
        cov_feat = self.cov_proj(Sigma_norm.reshape(-1))
        feat = self.feat_norm(x_t)
        combined = torch.cat([feat, cov_feat], dim=-1)
        return self.net(combined)


class TransformerPortfolioPolicy(nn.Module):
    """Self-attention portfolio policy inspired by Portfolio Transformer (Zhang et al.).

    Architecture: per-asset feature encoding → self-attention across assets → weight output.
    Unlike DHRP's tree routing, this uses flat global attention over all assets.
    Used in ablation A11 to compare attention vs tree for exploiting LLM features.
    """

    def __init__(self, feature_dim, n_assets, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.feature_dim = feature_dim
        self.d_model = d_model

        # Per-asset feature projection
        self.input_proj = nn.Linear(feature_dim + n_assets, d_model)
        self.pos_emb = nn.Parameter(torch.randn(n_assets, d_model) * 0.02)

        # Transformer encoder (self-attention across assets)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Weight head
        self.weight_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x_t, Sigma_t):
        """Forward pass matching DHRPLayer signature.

        Args:
            x_t: (feature_dim,) global feature vector
            Sigma_t: (n_assets, n_assets) covariance matrix
        Returns:
            (n_assets,) portfolio weights summing to 1
        """
        # Build per-asset tokens: [global_features || cov_row_i] for each asset
        cov_norm = Sigma_t / (Sigma_t.abs().max() + 1e-6)
        x_rep = x_t.unsqueeze(0).expand(self.n_assets, -1)  # (n_assets, feature_dim)
        tokens = torch.cat([x_rep, cov_norm], dim=-1)  # (n_assets, feature_dim + n_assets)

        # Project and add positional embedding
        h = self.input_proj(tokens) + self.pos_emb  # (n_assets, d_model)

        # Self-attention across assets
        h = self.encoder(h.unsqueeze(0)).squeeze(0)  # (n_assets, d_model)

        # Output weights
        logits = self.weight_head(h).squeeze(-1)  # (n_assets,)
        return F.softmax(logits, dim=-1)


class PPOPortfolioAgent(nn.Module):
    """PPO-style actor-critic for portfolio allocation (SAPPO @ ICML 2025 style).

    Actor: outputs Dirichlet concentration parameters for portfolio weights.
    Critic: estimates value function for baseline subtraction.
    Self-contained — no external RL library dependency.
    """

    def __init__(self, feature_dim, n_assets, hidden=128, dropout=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.feature_dim = feature_dim

        # Shared feature encoder
        cov_dim = n_assets * n_assets
        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.cov_proj = nn.Sequential(
            nn.Linear(cov_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )

        # Actor head: Dirichlet concentration parameters
        self.actor = nn.Sequential(
            nn.Linear(hidden, n_assets),
            nn.Softplus(),  # ensures positive concentrations
        )

        # Critic head: value function
        self.critic = nn.Linear(hidden, 1)

    def _encode(self, x_t, Sigma_t):
        Sigma_norm = Sigma_t / (Sigma_t.abs().max() + 1e-6)
        f = self.feat_proj(x_t)
        c = self.cov_proj(Sigma_norm.reshape(-1))
        return self.shared(torch.cat([f, c], dim=-1))

    def forward(self, x_t, Sigma_t):
        """Forward pass returning deterministic (mode) weights.

        Matches DHRPLayer signature for inference/backtest.
        """
        h = self._encode(x_t, Sigma_t)
        alpha = self.actor(h) + 1.0  # Dirichlet concentrations > 1 for mode
        # Dirichlet mode: (alpha_i - 1) / (sum(alpha) - K)
        weights = (alpha - 1) / ((alpha - 1).sum() + 1e-8)
        return torch.clamp(weights, min=1e-6) / torch.clamp(weights, min=1e-6).sum()

    def get_action_and_value(self, x_t, Sigma_t):
        """Sample portfolio weights from Dirichlet and compute value + log_prob.

        Used during PPO training (not during inference).
        """
        h = self._encode(x_t, Sigma_t)
        alpha = self.actor(h) + 1.0
        dist = torch.distributions.Dirichlet(alpha)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        value = self.critic(h).squeeze(-1)
        return action, log_prob, value

    def evaluate_actions(self, x_t, Sigma_t, actions):
        """Evaluate log_prob and value for given actions (PPO update step)."""
        h = self._encode(x_t, Sigma_t)
        alpha = self.actor(h) + 1.0
        dist = torch.distributions.Dirichlet(alpha)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(h).squeeze(-1)
        return log_prob, entropy, value


def train_ppo_agent(prices, device="cpu", is_em=False, volume=None, fdim=64,
                    epochs=40, lr=3e-4, gamma=0.99, clip_eps=0.2, n_steps=32,
                    train_end=None):
    """Train PPO portfolio agent on historical data.

    Uses rolling windows of returns as episodes. Reward = risk-adjusted return.
    """
    from ..data.feature_engineering import build_dataset, make_features
    import numpy as np

    X, S, R, H = build_dataset(prices, is_em=is_em, volume=volume, fdim=fdim,
                                train_end=train_end)
    if X.ndim == 1 or X.shape[0] < 50:
        raise ValueError(f"Insufficient data: {X.shape[0] if X.ndim > 1 else 0}")
    n_samp, fdim_actual, n_assets = X.shape[0], X.shape[1], prices.shape[1]

    agent = PPOPortfolioAgent(fdim_actual, n_assets).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    Xt = torch.from_numpy(X).to(device)
    St = torch.from_numpy(S).to(device)
    Rt = torch.from_numpy(R).to(device)

    best_reward, best_st = -float("inf"), None
    for ep in range(epochs):
        perm = torch.randperm(n_samp)
        ep_reward = 0.0
        n_batches = 0

        for start in range(0, n_samp - n_steps, n_steps):
            idx = perm[start : start + n_steps]

            # Collect rollout (detach actions/rewards for PPO — no graph needed)
            actions, log_probs, values, rewards = [], [], [], []
            with torch.no_grad():
                for i in idx:
                    a, lp, v = agent.get_action_and_value(Xt[i], St[i])
                    r = (a * Rt[i]).sum()
                    actions.append(a)
                    log_probs.append(lp)
                    values.append(v)
                    rewards.append(r)

            actions = torch.stack(actions).detach()
            old_log_probs = torch.stack(log_probs).detach()
            rewards_t = torch.stack(rewards).detach()

            # Compute advantages
            values_t = torch.stack(values).detach()
            advantages = (rewards_t - values_t)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update (K inner epochs)
            for _ in range(4):
                new_log_probs = []
                new_values = []
                new_entropies = []
                for k, i in enumerate(idx):
                    nlp, ent, nv = agent.evaluate_actions(Xt[i], St[i], actions[k])
                    new_log_probs.append(nlp)
                    new_values.append(nv)
                    new_entropies.append(ent)
                new_log_probs = torch.stack(new_log_probs)
                new_values = torch.stack(new_values)
                new_entropies = torch.stack(new_entropies)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, rewards_t.detach())
                entropy_bonus = -new_entropies.mean() * 0.01
                loss = policy_loss + 0.5 * value_loss + entropy_bonus

                optimizer.zero_grad()
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()

            ep_reward += rewards_t.mean().item()
            n_batches += 1

        if n_batches > 0:
            avg_reward = ep_reward / n_batches
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_st = {k: v.cpu().clone() for k, v in agent.state_dict().items()}
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  [PPO] Epoch {ep + 1}/{epochs}, avg_reward={avg_reward:.6f}")

    if best_st:
        agent.load_state_dict({k: v.to(device) for k, v in best_st.items()})
    return agent


def train_transformer_policy(prices, device="cpu", is_em=False, volume=None, fdim=64,
                             epochs=40, lr=3e-4, batch_size=32, train_end=None):
    """Train Transformer portfolio policy via direct Sharpe optimization."""
    from ..data.feature_engineering import build_dataset
    from ..models.loss_functions import dhrp_loss
    import numpy as np

    X, S, R, H = build_dataset(prices, is_em=is_em, volume=volume, fdim=fdim,
                                train_end=train_end)
    if X.ndim == 1 or X.shape[0] < 50:
        raise ValueError(f"Insufficient data: {X.shape[0] if X.ndim > 1 else 0}")
    n_samp, fdim_actual, n_assets = X.shape[0], X.shape[1], prices.shape[1]

    model = TransformerPortfolioPolicy(fdim_actual, n_assets).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)

    Xt = torch.from_numpy(X).to(device)
    St = torch.from_numpy(S).to(device)
    Rt = torch.from_numpy(R).to(device)

    best_loss, best_st = float("inf"), None
    for ep in range(epochs):
        perm = torch.randperm(n_samp)
        ep_loss, nb = 0.0, 0

        for s in range(0, n_samp, batch_size):
            e = min(s + batch_size, n_samp)
            opt.zero_grad()
            loss = dhrp_loss(
                model, Xt[perm[s:e]], St[perm[s:e]], Rt[perm[s:e]],
                H[perm[s:e].cpu().numpy()], is_em=is_em, lam_hrp=0.1,
            )
            if not torch.isnan(loss) and loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item()
                nb += 1
        sched.step()

        if nb > 0:
            avg = ep_loss / nb
            if avg < best_loss:
                best_loss = avg
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  [Transformer] Epoch {ep + 1}/{epochs}, loss={avg:.6f}")

    if best_st:
        model.load_state_dict({k: v.to(device) for k, v in best_st.items()})
    return model


class DFLPortfolioPolicy(nn.Module):
    """Decision-Focused Learning baseline (Wilder et al. 2019, Elmachtoub & Grigas 2022).

    Predicts expected returns and residual covariance, then solves mean-variance
    optimization differentiably via a soft-regularized closed-form solution.
    Trained end-to-end with a decision loss (Sharpe or CRRA), not MSE on returns.

    This is the hot trend at ICAIF 2025 — papers like "Return Prediction for
    Mean-Variance Portfolio Selection: How Decision-Focused Learning Shapes
    Forecasting Models" — a natural baseline to compete against.
    """

    def __init__(self, feature_dim, n_assets, hidden=128, dropout=0.1,
                 risk_aversion=2.0):
        super().__init__()
        self.n_assets = n_assets
        self.feature_dim = feature_dim
        self.risk_aversion = risk_aversion

        self.feat_norm = nn.LayerNorm(feature_dim)
        cov_dim = n_assets * n_assets
        self.cov_proj = nn.Sequential(
            nn.Linear(cov_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, feature_dim),
        )

        # Return prediction head
        self.return_head = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_assets),
        )

        # Covariance adjustment head (diagonal residual to input covariance)
        self.cov_adjust_head = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_assets),
            nn.Softplus(),  # positive diagonal adjustment
        )

        # Learnable regularization
        self.log_reg = nn.Parameter(torch.tensor(-2.0))  # exp(-2) ~ 0.135

    def forward(self, x_t, Sigma_t):
        """Forward pass: predict mu, adjust cov, solve MV with soft regularization.

        w = softmax(-gamma * argmin_w [w^T Sigma w - mu^T w + lambda ||w - 1/n||^2])

        Closed-form with diagonal augmentation:
            w_unnorm = (Sigma + lam*I)^(-1) mu
            w = softmax(gamma * w_unnorm)  # ensure non-negative and sum-to-1
        """
        Sigma_norm = Sigma_t / (Sigma_t.abs().max() + 1e-6)
        cov_feat = self.cov_proj(Sigma_norm.reshape(-1))
        feat = self.feat_norm(x_t)
        combined = torch.cat([feat, cov_feat], dim=-1)

        mu_pred = self.return_head(combined)          # (n_assets,)
        cov_diag_adj = self.cov_adjust_head(combined) # (n_assets,) positive
        lam = self.log_reg.exp() + 1e-4

        # Augmented covariance: add learned diagonal + regularization
        n = self.n_assets
        eye = torch.eye(n, device=x_t.device)
        Sigma_aug = Sigma_t + torch.diag(cov_diag_adj) + lam * eye

        # Differentiable MV solution via Cholesky solve
        try:
            L = torch.linalg.cholesky(Sigma_aug)
            w_raw = torch.cholesky_solve(mu_pred.unsqueeze(-1), L).squeeze(-1)
        except Exception:
            # Fallback: inverse with small ridge
            Sigma_ridge = Sigma_aug + 1e-3 * eye
            w_raw = torch.linalg.solve(Sigma_ridge, mu_pred)

        # Project to simplex via softmax (long-only, sum-to-1)
        w = F.softmax(self.risk_aversion * w_raw, dim=-1)
        return w


def train_dfl_baseline(prices, device="cpu", is_em=False, volume=None,
                       epochs=40, lr=3e-4, fdim=None, seed=42, train_end=None):
    """Train the Decision-Focused Learning baseline."""
    import numpy as np
    from ..data.feature_engineering import build_dataset, DEFAULT_FDIM
    from .loss_functions import dhrp_loss

    if fdim is None:
        fdim = DEFAULT_FDIM

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X, S, R, H = build_dataset(prices, is_em=is_em, volume=volume, fdim=fdim,
                                train_end=train_end)
    if X.ndim == 1 or X.shape[0] < 50:
        raise ValueError(f"Insufficient data: {X.shape[0] if X.ndim > 1 else 0}")

    n_samp, fdim_actual, n_assets = X.shape[0], X.shape[1], prices.shape[1]
    print(f"  [DFL] {n_samp} samples, {n_assets} assets, fdim={fdim_actual}")

    model = DFLPortfolioPolicy(fdim_actual, n_assets).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)

    Xt = torch.from_numpy(X).to(device)
    St = torch.from_numpy(S).to(device)
    Rt = torch.from_numpy(R).to(device)

    best_loss, best_st = float("inf"), None
    for ep in range(epochs):
        perm = torch.randperm(n_samp)
        ep_loss, nb = 0.0, 0

        for s in range(0, n_samp, 32):
            e = min(s + 32, n_samp)
            opt.zero_grad()
            loss = dhrp_loss(
                model, Xt[perm[s:e]], St[perm[s:e]], Rt[perm[s:e]],
                H[perm[s:e].cpu().numpy()], is_em=is_em, lam_hrp=0.1,
            )
            if not torch.isnan(loss) and loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item()
                nb += 1
        sched.step()

        if nb > 0:
            avg = ep_loss / nb
            if avg < best_loss:
                best_loss = avg
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  [DFL] Epoch {ep + 1}/{epochs}, loss={avg:.6f}")

    if best_st:
        model.load_state_dict({k: v.to(device) for k, v in best_st.items()})
    return model
