import numpy as np
import cvxpy as cp
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def equal_weight(mu, cov):
    return np.ones(len(mu)) / len(mu)


def min_variance(mu, cov, is_em=False):
    n = len(mu)
    w = cp.Variable(n)
    obj = cp.quad_form(w, cov)
    if is_em:
        obj += 0.01 * cp.sum_squares(w - 1 / n)
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(w) == 1, w >= 0])
    prob.solve(solver=cp.OSQP)
    return np.array(w.value).flatten()


def mean_variance(mu, cov, is_em=False):
    n = len(mu)
    w = cp.Variable(n)
    if is_em:
        obj = w @ mu - 0.3 * cp.quad_form(w, cov) - 0.02 * cp.sum_squares(w - 1 / n)
        prob = cp.Problem(cp.Maximize(obj), [cp.sum(w) == 1, w >= 0, w <= 0.4])
    else:
        obj = w @ mu - 0.5 * cp.quad_form(w, cov)
        prob = cp.Problem(cp.Maximize(obj), [cp.sum(w) == 1, w >= 0])
    prob.solve(solver=cp.OSQP)
    return np.array(w.value).flatten()


def hrp_allocation(cov):
    corr = np.corrcoef(cov) if cov.ndim == 2 else cov
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="ward"))

    def _bisect(ids):
        if len(ids) == 1:
            return {ids[0]: 1.0}
        mid = len(ids) // 2
        left, right = ids[:mid], ids[mid:]
        vl = np.sqrt(np.diag(cov[np.ix_(left, left)]).sum())
        vr = np.sqrt(np.diag(cov[np.ix_(right, right)]).sum())
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
        sigma_p = np.sqrt(w @ cov @ w)
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
    vols = np.sqrt(np.diag(cov))
    w = cp.Variable(n)
    # Maximize w^T sigma / sqrt(w^T Sigma w) is non-convex;
    # use the dual: minimize w^T Sigma w subject to w^T sigma = 1
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, cov)),
        [w @ vols == 1, w >= 0],
    )
    prob.solve(solver=cp.OSQP)
    raw = np.array(w.value).flatten()
    return raw / raw.sum()
