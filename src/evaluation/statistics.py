import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_rel
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac


def compute_sharpe(r, rf=0.03):
    """Compute annualized Sharpe ratio."""
    exc = r - rf / 252
    return (exc.mean() * 252) / (exc.std() * np.sqrt(252)) if exc.std() > 0 else 0


def compute_stats(results, rf=0.03, n_boot=1000):
    """Compute Sharpe ratio, HAC t-stat, and BCa bootstrap CI for each method."""
    stats = []
    for m in sorted(results["method"].unique()):
        r = results[results["method"] == m]["return"].dropna().values
        if len(r) == 0:
            continue
        exc = r - rf / 252
        vol = exc.std()
        sharpe = (exc.mean() * 252) / (vol * np.sqrt(252)) if vol > 0 else np.nan

        # HAC t-stat
        try:
            if len(exc) >= 30:
                mod = OLS(exc, add_constant(np.zeros(len(exc)))).fit()
                hac_t = (exc.mean() * 252) / np.sqrt(cov_hac(mod)[0, 0] * 252)
            else:
                hac_t = np.nan
        except Exception:
            hac_t = np.nan

        # BCa (bias-corrected accelerated) bootstrap CI
        ci_lo, ci_hi = _bca_bootstrap_ci(r, rf=rf, n_boot=n_boot)

        # Max drawdown
        cum = (1 + pd.Series(r)).cumprod()
        peak = cum.cummax()
        mdd = ((cum - peak) / peak).min()

        # Calmar ratio
        calmar = (sharpe * vol * np.sqrt(252)) / abs(mdd) if abs(mdd) > 1e-8 else np.nan

        stats.append({
            "Method": m, "Sharpe": sharpe, "HAC_t": hac_t,
            "CI_lo": ci_lo, "CI_hi": ci_hi,
            "MaxDD": mdd, "Calmar": calmar,
        })
    return pd.DataFrame(stats)


def _bca_bootstrap_ci(r, rf=0.03, n_boot=1000, alpha=0.05):
    """BCa (bias-corrected accelerated) bootstrap confidence interval for Sharpe."""
    n = len(r)
    if n < 30:
        return np.nan, np.nan

    observed = compute_sharpe(r, rf)

    # Bootstrap distribution
    np.random.seed(42)
    boot_sharpes = []
    bl = max(5, int(n ** (1 / 3)))  # block length for block bootstrap
    for _ in range(n_boot):
        idx = np.random.randint(0, n - bl + 1, n // bl + 1)
        br = np.concatenate([r[i : i + bl] for i in idx])[:n]
        exc = br - rf / 252
        if exc.std() > 0:
            boot_sharpes.append((exc.mean() * 252) / (exc.std() * np.sqrt(252)))
    if len(boot_sharpes) < 50:
        return np.nan, np.nan
    boot_sharpes = np.array(boot_sharpes)

    # Bias correction factor (z0)
    z0 = norm.ppf(np.mean(boot_sharpes < observed))

    # Acceleration factor (a) via jackknife
    jk_sharpes = np.zeros(n)
    for i in range(n):
        ri = np.delete(r, i)
        exc = ri - rf / 252
        jk_sharpes[i] = (exc.mean() * 252) / (exc.std() * np.sqrt(252)) if exc.std() > 0 else 0
    jk_mean = jk_sharpes.mean()
    num = np.sum((jk_mean - jk_sharpes) ** 3)
    den = 6 * (np.sum((jk_mean - jk_sharpes) ** 2) ** 1.5)
    a = num / den if abs(den) > 1e-12 else 0

    # Adjusted percentiles
    z_alpha = norm.ppf(alpha / 2)
    z_1alpha = norm.ppf(1 - alpha / 2)

    p_lo = norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    p_hi = norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    p_lo = np.clip(p_lo, 0.5 / n_boot, 1 - 0.5 / n_boot)
    p_hi = np.clip(p_hi, 0.5 / n_boot, 1 - 0.5 / n_boot)

    return float(np.percentile(boot_sharpes, p_lo * 100)), float(np.percentile(boot_sharpes, p_hi * 100))


def sharpe_difference_test(results, method_a="DHRP", method_b="HRP", n_boot=1000, rf=0.03):
    """Bootstrap test for Sharpe ratio difference between two methods."""
    ser_a = results[results["method"] == method_a].set_index("date")["return"]
    ser_b = results[results["method"] == method_b].set_index("date")["return"]
    merged = pd.concat([ser_a.rename("A"), ser_b.rename("B")], axis=1).dropna()

    r_a, r_b = merged["A"].values, merged["B"].values
    n = len(r_a)
    actual_diff = compute_sharpe(r_a, rf) - compute_sharpe(r_b, rf)

    # Paired t-test
    t_paired, p_paired = ttest_rel(r_a, r_b)

    # Bootstrap
    np.random.seed(42)
    sharpe_diffs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        sr_a = compute_sharpe(r_a[idx], rf)
        sr_b = compute_sharpe(r_b[idx], rf)
        sharpe_diffs.append(sr_a - sr_b)
    sharpe_diffs = np.array(sharpe_diffs)
    ci_lo, ci_hi = np.percentile(sharpe_diffs, [2.5, 97.5])
    p_boot = np.mean(sharpe_diffs < 0) * 2  # two-sided

    # Cohen's d effect size
    diff_returns = r_a - r_b
    cohens_d = diff_returns.mean() / (diff_returns.std() + 1e-12)

    # Information ratio of the difference
    diff = r_a - r_b
    ir = (diff.mean() * 252) / (diff.std() * np.sqrt(252)) if diff.std() > 0 else 0

    return {
        "method_a": method_a,
        "method_b": method_b,
        "sharpe_a": compute_sharpe(r_a, rf),
        "sharpe_b": compute_sharpe(r_b, rf),
        "sharpe_diff": actual_diff,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "bootstrap_se": sharpe_diffs.std(),
        "bootstrap_p": p_boot,
        "paired_t": t_paired,
        "paired_p": p_paired,
        "information_ratio": ir,
        "cohens_d": cohens_d,
    }


def fdr_correct(p_values, method="bh"):
    """Benjamini-Hochberg FDR correction for multiple testing.

    Args:
        p_values: list or array of raw p-values
        method: "bh" for Benjamini-Hochberg, "bonferroni" for Bonferroni
    Returns:
        dict with 'raw', 'adjusted', and 'significant' arrays
    """
    p = np.array(p_values)
    n = len(p)
    if n == 0:
        return {"raw": p, "adjusted": p, "significant": np.array([], dtype=bool)}

    if method == "bonferroni":
        adjusted = np.minimum(p * n, 1.0)
    else:  # Benjamini-Hochberg
        sorted_idx = np.argsort(p)
        sorted_p = p[sorted_idx]
        adjusted = np.zeros(n)
        for i in range(n):
            adjusted[sorted_idx[i]] = sorted_p[i] * n / (i + 1)
        # Enforce monotonicity
        for i in range(n - 2, -1, -1):
            adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i]], adjusted[sorted_idx[i + 1]] if i + 1 < n else 1.0)
        adjusted = np.minimum(adjusted, 1.0)

    return {
        "raw": p,
        "adjusted": adjusted,
        "significant_005": adjusted < 0.05,
        "significant_010": adjusted < 0.10,
    }


def diebold_mariano_test(results, method_a="DHRP", method_b="HRP", loss_fn="squared"):
    """Diebold-Mariano test for equal predictive accuracy."""
    ser_a = results[results["method"] == method_a].set_index("date")["return"]
    ser_b = results[results["method"] == method_b].set_index("date")["return"]
    merged = pd.concat([ser_a.rename("A"), ser_b.rename("B")], axis=1).dropna()

    if loss_fn == "squared":
        # Use squared returns as loss (lower = better risk management)
        e_a = merged["A"].values ** 2
        e_b = merged["B"].values ** 2
    else:
        # Use negative returns as loss (higher return = lower loss)
        e_a = -merged["A"].values
        e_b = -merged["B"].values

    d = e_a - e_b
    T = len(d)
    d_mean = d.mean()

    # Newey-West variance with 1 lag
    gamma_0 = np.var(d, ddof=1)
    if T > 1:
        gamma_1 = np.cov(d[1:], d[:-1])[0, 1]
        V = gamma_0 + 2 * gamma_1
    else:
        V = gamma_0
    V = max(V, 1e-12)

    DM = d_mean / np.sqrt(V / T)
    p_value = 2 * (1 - norm.cdf(abs(DM)))
    return {"DM_stat": DM, "p_value": p_value, "loss_fn": loss_fn}


def subperiod_analysis(results, rf=0.03):
    """Compute Sharpe ratios per sub-period (pre-COVID, COVID, post-COVID).

    Returns DataFrame with columns [Method, period, Sharpe, n_obs].
    """
    periods = {
        "Pre-COVID (2016-2019)": ("2016-01-01", "2019-12-31"),
        "COVID (2020-2021)": ("2020-01-01", "2021-12-31"),
        "Post-COVID (2022+)": ("2022-01-01", "2099-12-31"),
    }
    rows = []
    for m in sorted(results["method"].unique()):
        ser = results[results["method"] == m].set_index("date")["return"]
        ser.index = pd.to_datetime(ser.index)
        for pname, (pstart, pend) in periods.items():
            sub = ser[(ser.index >= pstart) & (ser.index <= pend)]
            if len(sub) >= 30:
                sharpe = compute_sharpe(sub.values, rf)
                rows.append({"Method": m, "Period": pname, "Sharpe": sharpe, "N": len(sub)})
    return pd.DataFrame(rows)
