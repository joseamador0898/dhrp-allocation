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


def hrp_allocation(cov):
    corr = _cov_to_corr(cov)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
    np.fill_diagonal(dist, 0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="ward"))

    def _bisect(ids):
        if len(ids) == 1:
            return {ids[0]: 1.0}
        mid = len(ids) // 2
        left, right = ids[:mid], ids[mid:]
        vl = np.sqrt(max(np.diag(cov[np.ix_(left, left)]).sum(), 1e-12))
        vr = np.sqrt(max(np.diag(cov[np.ix_(right, right)]).sum(), 1e-12))
        wl = vr / (vl + vr)
        return {
            **{k: v * wl for k, v in _bisect(left).items()},
            **{k: v * (1 - wl) for k, v in _bisect(right).items()},
        }

    return np.array([_bisect(list(order)).get(i, 0) for i in range(len(cov))])


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
