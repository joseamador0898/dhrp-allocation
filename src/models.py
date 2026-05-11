"""src.models (consolidated). Original layout: ['baselines.py', 'deep_baselines.py', 'dhrp_layer.py', 'llm_dhrp_layer.py', 'loss_functions.py']"""

# ====================================================================
# Module: baselines.py
# ====================================================================
import numpy as np
import cvxpy as cp
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


def _solve_robust(prob, n):
    """Solve CVXPY problem with solver fallback chain."""
    for solver in [cp.SCS, cp.ECOS, cp.OSQP]:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate") and prob.variables()[0].value is not None:
                w = np.array(prob.variables()[0].value).flatten()
                if not np.any(np.isnan(w)):
                    return w
        except (cp.SolverError, Exception):
            continue
    return np.ones(n) / n


def equal_weight(mu, cov):
    return np.ones(len(mu)) / len(mu)


def min_variance(mu, cov, is_em=False):
    n = len(mu)
    # Regularize covariance for numerical stability
    cov_reg = cov + np.eye(n) * (1e-4 * np.trace(cov) / n)
    cov_reg = (cov_reg + cov_reg.T) / 2  # ensure symmetry
    w = cp.Variable(n)
    obj = cp.quad_form(w, cp.psd_wrap(cov_reg))
    if is_em:
        obj += 0.01 * cp.sum_squares(w - 1 / n)
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(w) == 1, w >= 0])
    return _solve_robust(prob, n)


def mean_variance(mu, cov, is_em=False):
    n = len(mu)
    cov_reg = cov + np.eye(n) * (1e-4 * np.trace(cov) / n)
    cov_reg = (cov_reg + cov_reg.T) / 2
    w = cp.Variable(n)
    if is_em:
        obj = w @ mu - 0.3 * cp.quad_form(w, cp.psd_wrap(cov_reg)) - 0.02 * cp.sum_squares(w - 1 / n)
        prob = cp.Problem(cp.Maximize(obj), [cp.sum(w) == 1, w >= 0, w <= 0.4])
    else:
        obj = w @ mu - 0.5 * cp.quad_form(w, cp.psd_wrap(cov_reg))
        prob = cp.Problem(cp.Maximize(obj), [cp.sum(w) == 1, w >= 0])
    return _solve_robust(prob, n)


def _cov_to_corr(cov):
    """Convert covariance matrix to correlation matrix properly."""
    d = np.sqrt(np.diag(cov))
    d = np.where(d < 1e-10, 1e-10, d)
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1, 1)


def hrp_allocation(cov, linkage_method="single"):
    """Classical long-only HRP allocation with inverse-variance bisection."""
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be a square matrix; got {cov.shape}")
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)

    eps = 1e-12
    cov = (cov + cov.T) / 2.0
    cov = cov + np.eye(n) * eps
    corr = _cov_to_corr(cov)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), method=linkage_method)))

    def _ivp(sub_cov):
        diag = np.maximum(np.diag(sub_cov), eps)
        inv_diag = 1.0 / diag
        return inv_diag / inv_diag.sum()

    def _cluster_var(ids):
        sub_cov = cov[np.ix_(ids, ids)]
        w = _ivp(sub_cov)
        return float(w @ sub_cov @ w)

    weights = {i: 1.0 for i in order}
    clusters = [order]
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            var_l = _cluster_var(left)
            var_r = _cluster_var(right)
            alpha = 1.0 - var_l / (var_l + var_r + eps)
            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= 1.0 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters

    out = np.array([weights[i] for i in range(n)], dtype=float)
    out = np.clip(out, 0.0, np.inf)
    return out / out.sum() if out.sum() > 0 else np.ones(n) / n


def risk_parity(cov, tol=1e-8, max_iter=500):
    """Equal risk contribution portfolio."""
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        sigma_p = np.sqrt(max(w @ cov @ w, 1e-16))
        rc = w * (cov @ w) / sigma_p
        target = sigma_p / n
        w_new = w * (target / (rc + 1e-12))
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            break
        w = w_new
    return w


def max_diversification(mu, cov):
    """Maximize diversification ratio: sum(w_i * sigma_i) / sigma_p."""
    n = len(mu)
    cov_reg = cov + np.eye(n) * (1e-4 * np.trace(cov) / n)
    cov_reg = (cov_reg + cov_reg.T) / 2
    vols = np.sqrt(np.diag(cov_reg))
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov_reg))),
        [w @ vols == 1, w >= 0],
    )
    raw = _solve_robust(prob, n)
    return raw / raw.sum()


def ledoit_wolf_cov(returns):
    """Compute Ledoit-Wolf shrinkage covariance estimate (annualized).

    Args:
        returns: DataFrame or ndarray of daily returns (T x n_assets)
    Returns:
        Annualized shrinkage covariance matrix (n_assets x n_assets)
    """
    if hasattr(returns, 'values'):
        returns = returns.values
    lw = LedoitWolf().fit(returns)
    return lw.covariance_ * 252

# ====================================================================
# Module: deep_baselines.py
# ====================================================================
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
    from src.data import build_dataset
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
    from src.data import build_dataset
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
    from src.data import build_dataset, DEFAULT_FDIM

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

# ====================================================================
# Module: dhrp_layer.py
# ====================================================================
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

    def get_gating_probs(self, x_t, Sigma_t, text_emb=None, macro_feat=None):
        """Return per-node gating probabilities for interpretability analysis.

        text_emb and macro_feat are accepted for API compatibility with
        LLMDHRPLayer but ignored here (DHRPLayer uses price features only).
        """
        Sigma_norm = Sigma_t / (Sigma_t.abs().max() + 1e-6)
        node_feat = self.feat_norm(x_t + self.cov_proj(Sigma_norm.reshape(-1)))
        temp = torch.clamp(self.log_temp.exp(), 0.1, 5.0)
        return [F.softmax(g(node_feat) / temp, dim=-1).detach() for g in self.gates]

# ====================================================================
# Module: llm_dhrp_layer.py
# ====================================================================
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
            gate_val = torch.sigmoid(self.fusion.gate_proj(torch.cat([price_feat, attn_out], dim=-1)))
            return {
                "gate_mean": gate_val.mean().item(),
                "gate_std": gate_val.std().item(),
                "attn_weights": attn_weights.detach() if attn_weights is not None else None,
            }
        return None

# ====================================================================
# Module: loss_functions.py
# ====================================================================
import torch


def dhrp_loss(layer, xb, Sb, rb, hrp_w, is_em=False, lam_hrp=0.3,
              prev_weights=None, lam_turnover=0.0):
    """Multi-objective loss: CRRA utility + Sharpe + HRP regularization + turnover penalty."""
    port_r, wts = [], []
    for t in range(rb.shape[0]):
        w = layer(xb[t], Sb[t])
        wts.append(w)
        port_r.append((w * rb[t]).sum())
    port_r = torch.stack(port_r)
    wts = torch.stack(wts)

    gamma = 1.2 if is_em else 2.5
    crra = (
        (torch.clamp(1 + port_r, min=0.1) ** (1 - gamma) - 1) / (1 - gamma)
    ).mean()

    sharpe = port_r.mean() / (port_r.std() + 1e-6) * (0.5 if is_em else 1.0)

    hrp_target = torch.from_numpy(hrp_w).to(xb.device).float()
    hrp_reg = ((wts - hrp_target) ** 2).mean() * lam_hrp * (0.5 if is_em else 0.2)

    risk = (
        torch.stack([wts[t] @ Sb[t] @ wts[t] for t in range(rb.shape[0])]).mean()
        * (0.004 if is_em else 0.001)
    )

    entropy = (
        (-(wts * torch.log(wts + 1e-8)).sum(1).mean() * 0.15) if is_em else 0
    )

    hhi = (wts ** 2).sum(1).mean()
    concentration_pen = hhi * (0.3 if is_em else 0.1)

    # Turnover penalty: penalize large weight changes between consecutive samples
    turnover_pen = 0.0
    if lam_turnover > 0:
        if prev_weights is not None:
            turnover_pen = torch.mean(torch.abs(wts - prev_weights)) * lam_turnover
        # Also penalize within-batch consecutive weight changes
        if wts.shape[0] > 1:
            intra_turnover = torch.mean(torch.abs(wts[1:] - wts[:-1])) * lam_turnover * 0.5
            turnover_pen = turnover_pen + intra_turnover

    loss = -(crra + sharpe + entropy) + hrp_reg + risk + concentration_pen + turnover_pen
    if torch.isnan(loss):
        return torch.tensor(0.0, device=xb.device, requires_grad=True)
    return loss

