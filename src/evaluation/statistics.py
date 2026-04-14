import time

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


def compute_stats(results, rf=0.03, n_boot=1000, gamma=2.5, oos_start=None):
    """Compute comprehensive performance metrics for each method.

    If ``oos_start`` is provided, restrict metrics to dates >= oos_start so
    headline numbers reflect strict out-of-sample performance.

    Returns DataFrame with columns: Method, Sharpe, Sortino, Calmar, MaxDD,
    CVaR_5, VaR_5, Omega, CER, Ann_Return, Ann_Vol, HAC_t, CI_lo, CI_hi.
    """
    if oos_start is not None and "date" in results.columns and not results.empty:
        cutoff = pd.Timestamp(oos_start)
        results = results[pd.to_datetime(results["date"]) >= cutoff]
    # Defensive dedup: if rolling_backtest ever emits duplicate (method, date)
    # rows (overlapping rebalance windows), average them so Sharpe /
    # Ann_Return / MaxDD / cumulative plots aren't multiply-counted.
    if "date" in results.columns and not results.empty:
        dup_mask = results.duplicated(subset=["method", "date"], keep=False)
        if dup_mask.any():
            import warnings
            n_dup = int(dup_mask.sum())
            warnings.warn(
                f"compute_stats: {n_dup} duplicate (method, date) rows detected; "
                "averaging. This usually means rolling_backtest was run with "
                "step_days < test_days. Expected to be zero after the "
                "backtest.py fix — investigate if you see this.",
                stacklevel=2,
            )
            results = (
                results.groupby(["method", "date"], as_index=False)["return"].mean()
            )

    stats = []
    for m in sorted(results["method"].unique()):
        r = (
            results[results["method"] == m]
            .sort_values("date")["return"] if "date" in results.columns
            else results[results["method"] == m]["return"]
        ).dropna().values
        if len(r) == 0:
            continue
        exc = r - rf / 252
        vol = exc.std()
        ann_ret = exc.mean() * 252
        ann_vol = vol * np.sqrt(252) if vol > 0 else np.nan
        sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan

        # HAC t-stat
        try:
            if len(exc) >= 30:
                mod = OLS(exc, add_constant(np.zeros(len(exc)))).fit()
                hac_t = (exc.mean() * 252) / np.sqrt(cov_hac(mod)[0, 0] * 252)
            else:
                hac_t = np.nan
        except Exception:
            hac_t = np.nan

        # BCa bootstrap CI
        ci_lo, ci_hi = _bca_bootstrap_ci(r, rf=rf, n_boot=n_boot)

        # Max drawdown
        cum = (1 + pd.Series(r)).cumprod()
        peak = cum.cummax()
        mdd = ((cum - peak) / peak).min()

        # Calmar ratio
        calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-8 else np.nan

        # Sortino ratio (downside deviation)
        downside = exc[exc < 0]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 1 else np.nan
        sortino = ann_ret / downside_std if downside_std and downside_std > 0 else np.nan

        # VaR and CVaR at 5%
        var_5 = float(np.percentile(r, 5))
        cvar_5 = float(r[r <= var_5].mean()) if np.any(r <= var_5) else var_5

        # Omega ratio (threshold = 0)
        gains = r[r > 0].sum()
        losses = abs(r[r <= 0].sum())
        omega = gains / losses if losses > 1e-12 else np.nan

        # Certainty Equivalent Return (CER) under CRRA utility
        try:
            u_vals = ((1 + np.clip(r, -0.99, None)) ** (1 - gamma)) / (1 - gamma)
            eu = u_vals.mean()
            cer = (eu * (1 - gamma)) ** (1 / (1 - gamma)) - 1
            cer_ann = cer * 252
        except Exception:
            cer_ann = np.nan

        stats.append({
            "Method": m, "Sharpe": sharpe, "Sortino": sortino,
            "Calmar": calmar, "MaxDD": mdd,
            "CVaR_5": cvar_5, "VaR_5": var_5, "Omega": omega,
            "CER": cer_ann, "Ann_Return": ann_ret, "Ann_Vol": ann_vol,
            "HAC_t": hac_t, "CI_lo": ci_lo, "CI_hi": ci_hi,
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


def jobson_korkie_test(r_a, r_b, rf=0.03):
    """Jobson-Korkie (1981) test for Sharpe ratio equality with
    Memmel (2003) correction. Returns (z_stat, p_value).

    More powerful than naive bootstrap for moderate-N samples.
    """
    n = len(r_a)
    if n < 30:
        return np.nan, np.nan
    exc_a = r_a - rf / 252
    exc_b = r_b - rf / 252
    mu_a, mu_b = exc_a.mean(), exc_b.mean()
    sd_a, sd_b = exc_a.std(ddof=1), exc_b.std(ddof=1)
    if sd_a < 1e-12 or sd_b < 1e-12:
        return np.nan, np.nan
    sr_a, sr_b = mu_a / sd_a, mu_b / sd_b
    rho = np.corrcoef(exc_a, exc_b)[0, 1]
    # Memmel-corrected variance
    var = (1.0 / n) * (
        2 * (1 - rho)
        + 0.5 * (sr_a ** 2 + sr_b ** 2 - 2 * sr_a * sr_b * rho ** 2)
    )
    if var <= 0:
        return np.nan, np.nan
    z = (sr_a - sr_b) / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


def sharpe_difference_test(results, method_a="DHRP", method_b="HRP", n_boot=1000, rf=0.03):
    """Test for Sharpe ratio difference between two methods.

    Combines: (1) Jobson-Korkie HAC-corrected parametric test (Memmel 2003),
    (2) stationary block bootstrap with proper centered p-value.
    """
    ser_a = results[results["method"] == method_a].set_index("date")["return"]
    ser_b = results[results["method"] == method_b].set_index("date")["return"]
    merged = pd.concat([ser_a.rename("A"), ser_b.rename("B")], axis=1).dropna()

    r_a, r_b = merged["A"].values, merged["B"].values
    n = len(r_a)
    sr_a = compute_sharpe(r_a, rf)
    sr_b = compute_sharpe(r_b, rf)
    actual_diff = sr_a - sr_b

    # Paired t-test on returns
    t_paired, p_paired = ttest_rel(r_a, r_b)

    # Jobson-Korkie / Memmel parametric test
    jk_z, jk_p = jobson_korkie_test(r_a, r_b, rf=rf)

    # Stationary block bootstrap on paired observations (preserves correlation)
    np.random.seed(42)
    bl = max(5, int(n ** (1 / 3)))
    sharpe_diffs = []
    for _ in range(n_boot):
        # Block bootstrap indices
        n_blocks = n // bl + 1
        starts = np.random.randint(0, n - bl + 1, n_blocks)
        idx = np.concatenate([np.arange(s, s + bl) for s in starts])[:n]
        sr_a_b = compute_sharpe(r_a[idx], rf)
        sr_b_b = compute_sharpe(r_b[idx], rf)
        sharpe_diffs.append(sr_a_b - sr_b_b)
    sharpe_diffs = np.array(sharpe_diffs)
    ci_lo, ci_hi = np.percentile(sharpe_diffs, [2.5, 97.5])

    # Correct two-sided bootstrap p-value: center distribution at H0 (=0)
    # then compute fraction at least as extreme as observed
    centered = sharpe_diffs - sharpe_diffs.mean()
    p_boot = float(np.mean(np.abs(centered) >= abs(actual_diff)))
    p_boot = max(p_boot, 1.0 / n_boot)  # never report exact 0

    # Use the more powerful of JK and bootstrap as the primary p-value
    # (both are valid; JK is parametric and tighter for clean returns)
    primary_p = jk_p if not np.isnan(jk_p) else p_boot

    # Cohen's d effect size
    diff_returns = r_a - r_b
    cohens_d = diff_returns.mean() / (diff_returns.std() + 1e-12)

    # Information ratio of the difference
    ir = (diff_returns.mean() * 252) / (diff_returns.std() * np.sqrt(252)) if diff_returns.std() > 0 else 0

    return {
        "method_a": method_a,
        "method_b": method_b,
        "sharpe_a": sr_a,
        "sharpe_b": sr_b,
        "sharpe_diff": actual_diff,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "bootstrap_se": sharpe_diffs.std(),
        "bootstrap_p": p_boot,
        "jk_z": jk_z,
        "jk_p": jk_p,
        "primary_p": primary_p,
        "paired_t": t_paired,
        "paired_p": p_paired,
        "information_ratio": ir,
        "cohens_d": cohens_d,
    }


def fdr_correct(p_values, method="bh"):
    """Multiple testing correction.

    Args:
        p_values: list or array of raw p-values
        method: "bh" for Benjamini-Hochberg, "bonferroni", or "holm" for Holm-Bonferroni
    Returns:
        dict with 'raw', 'adjusted', and 'significant' arrays
    """
    p = np.array(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return {"raw": p, "adjusted": p, "significant_005": np.array([], dtype=bool),
                "significant_010": np.array([], dtype=bool)}

    if method == "bonferroni":
        adjusted = np.minimum(p * n, 1.0)
    elif method == "holm":
        # Holm-Bonferroni: step-down procedure controlling FWER
        sorted_idx = np.argsort(p)
        adjusted = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            adjusted[idx] = p[idx] * (n - i)
        # Enforce monotonicity (step-down)
        running_max = 0.0
        for i in range(n):
            idx = sorted_idx[i]
            adjusted[idx] = max(adjusted[idx], running_max)
            running_max = adjusted[idx]
        adjusted = np.minimum(adjusted, 1.0)
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


# ---------------------------------------------------------------------------
# Pairwise statistical tests with multiple testing correction
# ---------------------------------------------------------------------------

def pairwise_sharpe_tests(results, ref="DHRP", n_boot=1000, rf=0.03, correction="holm"):
    """Run pairwise Sharpe difference tests between ref and all other methods.

    Uses Jobson-Korkie/Memmel parametric test as primary (more powerful than
    naive bootstrap for moderate samples), with Holm-Bonferroni correction.
    """
    methods = sorted(results["method"].unique())
    if ref not in methods:
        return pd.DataFrame()
    others = [m for m in methods if m != ref]
    rows = []
    raw_ps = []
    for other in others:
        try:
            res = sharpe_difference_test(results, method_a=ref, method_b=other,
                                         n_boot=n_boot, rf=rf)
            rows.append(res)
            # Prefer Jobson-Korkie (more powerful) over naive bootstrap
            raw_ps.append(res.get("primary_p", res["bootstrap_p"]))
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    corrected = fdr_correct(raw_ps, method=correction)
    df = pd.DataFrame(rows)
    df["adjusted_p"] = corrected["adjusted"]
    df["significant_005"] = corrected["significant_005"]
    df["significant_010"] = corrected["significant_010"]
    return df


def pairwise_dm_tests(results, ref="DHRP", loss_fn="squared", correction="holm"):
    """Run pairwise Diebold-Mariano tests between ref and all other methods.

    Returns DataFrame with raw and corrected p-values.
    """
    methods = sorted(results["method"].unique())
    if ref not in methods:
        return pd.DataFrame()
    others = [m for m in methods if m != ref]
    rows = []
    raw_ps = []
    for other in others:
        try:
            res = diebold_mariano_test(results, method_a=ref, method_b=other,
                                       loss_fn=loss_fn)
            res["method_a"] = ref
            res["method_b"] = other
            rows.append(res)
            raw_ps.append(res["p_value"])
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    corrected = fdr_correct(raw_ps, method=correction)
    df = pd.DataFrame(rows)
    df["adjusted_p"] = corrected["adjusted"]
    df["significant_005"] = corrected["significant_005"]
    return df


def full_statistical_battery(results, ref="DHRP", rf=0.03, n_boot=1000):
    """Run comprehensive statistical tests. Returns dict of DataFrames."""
    return {
        "sharpe_tests": pairwise_sharpe_tests(results, ref=ref, n_boot=n_boot, rf=rf),
        "dm_tests_squared": pairwise_dm_tests(results, ref=ref, loss_fn="squared"),
        "dm_tests_negative": pairwise_dm_tests(results, ref=ref, loss_fn="negative"),
        "spa": spa_test(results, benchmark="EW", n_boot=n_boot, rf=rf),
        "mcs": model_confidence_set(results, alpha=0.05, n_boot=n_boot, rf=rf),
    }


# ---------------------------------------------------------------------------
# Regime / sub-period analysis
# ---------------------------------------------------------------------------

def subperiod_analysis(results, rf=0.03, gamma=2.5, train_end=None):
    """Compute comprehensive metrics per macro regime.

    When ``train_end`` is provided, always emit an "OOS Full" row covering
    ``train_end`` through the end of the data so the OOS window is visible
    even if none of the legacy regime buckets line up with the split.

    Returns DataFrame with columns [Method, Period, Sharpe, Sortino, MaxDD, CVaR_5, N].
    """
    periods = {
        "Pre-COVID Bull (2016-2019)": ("2016-01-01", "2019-12-31"),
        "COVID Crash (2020-H1)": ("2020-01-01", "2020-06-30"),
        "Recovery (2020-H2 to 2021)": ("2020-07-01", "2021-12-31"),
        "Rate Hikes (2022 to 2023-H1)": ("2022-01-01", "2023-06-30"),
        "Post-Hike (2023-H2+)": ("2023-07-01", "2099-12-31"),
    }
    if train_end is not None:
        periods[f"OOS Full ({pd.Timestamp(train_end).strftime('%Y-%m-%d')}+)"] = (
            pd.Timestamp(train_end).strftime("%Y-%m-%d"), "2099-12-31",
        )
    rows = []
    for m in sorted(results["method"].unique()):
        ser = results[results["method"] == m].set_index("date")["return"]
        ser.index = pd.to_datetime(ser.index)
        for pname, (pstart, pend) in periods.items():
            sub = ser[(ser.index >= pstart) & (ser.index <= pend)]
            if len(sub) >= 20:
                r = sub.values
                exc = r - rf / 252
                vol = exc.std()
                ann_ret = exc.mean() * 252
                ann_vol = vol * np.sqrt(252) if vol > 0 else np.nan
                sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan

                # Sortino
                down = exc[exc < 0]
                down_std = down.std() * np.sqrt(252) if len(down) > 1 else np.nan
                sortino = ann_ret / down_std if down_std and down_std > 0 else np.nan

                # MaxDD
                cum = (1 + pd.Series(r)).cumprod()
                mdd = ((cum - cum.cummax()) / cum.cummax()).min()

                # CVaR
                var5 = np.percentile(r, 5)
                cvar5 = float(r[r <= var5].mean()) if np.any(r <= var5) else var5

                rows.append({
                    "Method": m, "Period": pname, "Sharpe": sharpe,
                    "Sortino": sortino, "MaxDD": mdd, "CVaR_5": cvar5,
                    "N": len(sub),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Transaction cost sensitivity
# ---------------------------------------------------------------------------

def compute_turnover(weights_history):
    """Compute average turnover from weight history.

    Args:
        weights_history: list of dicts with keys [method, date, weights]
    Returns:
        DataFrame with [Method, Avg_Turnover, Total_Turnover, N_Rebalances]
    """
    if not weights_history:
        return pd.DataFrame()
    df = pd.DataFrame(weights_history)
    rows = []
    for m in sorted(df["method"].unique()):
        mw = df[df["method"] == m].sort_values("date")
        w_arr = np.array(mw["weights"].tolist())
        if len(w_arr) < 2:
            continue
        turnovers = np.sum(np.abs(np.diff(w_arr, axis=0)), axis=1)
        rows.append({
            "Method": m,
            "Avg_Turnover": turnovers.mean(),
            "Total_Turnover": turnovers.sum(),
            "N_Rebalances": len(turnovers),
        })
    return pd.DataFrame(rows)


def cost_sensitivity_analysis(results, weights_history, cost_levels=None, rf=0.03):
    """Compute net-of-cost Sharpe at various transaction cost levels.

    Args:
        results: backtest results DataFrame
        weights_history: list of dicts with [method, date, weights]
        cost_levels: list of costs in bps (default [0, 5, 10, 20, 50])
    Returns:
        DataFrame with columns [Method, cost_0bps, cost_5bps, ...]
    """
    if cost_levels is None:
        cost_levels = [0, 5, 10, 20, 50]
    if not weights_history:
        return pd.DataFrame()

    wh_df = pd.DataFrame(weights_history)
    methods = sorted(results["method"].unique())
    rows = []

    for m in methods:
        r = results[results["method"] == m].sort_values("date")
        mw = wh_df[wh_df["method"] == m].sort_values("date")
        if mw.empty or r.empty:
            continue

        base_returns = r["return"].values.copy()
        w_arr = np.array(mw["weights"].tolist())

        # Compute turnover per rebalance
        turnovers = np.zeros(len(w_arr))
        if len(w_arr) > 1:
            turnovers[1:] = np.sum(np.abs(np.diff(w_arr, axis=0)), axis=1)

        # Map rebalance turnovers to daily returns
        rebal_dates = mw["date"].values
        daily_dates = r["date"].values

        row = {"Method": m}
        for cost_bps in cost_levels:
            adj_returns = base_returns.copy()
            # Spread turnover cost across holding period days
            for i, rd in enumerate(rebal_dates):
                mask = daily_dates == rd
                if mask.any():
                    tc = turnovers[i] * cost_bps / 10000
                    idx = np.where(mask)[0][0]
                    adj_returns[idx] -= tc
            exc = adj_returns - rf / 252
            vol = exc.std()
            s = (exc.mean() * 252) / (vol * np.sqrt(252)) if vol > 0 else np.nan
            row[f"cost_{cost_bps}bps"] = s
        rows.append(row)

    return pd.DataFrame(rows)


def breakeven_cost(results, weights_history, method_a="DHRP", method_b="HRP",
                   rf=0.03, max_bps=200):
    """Find the transaction cost level at which method_a's Sharpe equals method_b's."""
    for bps in range(0, max_bps + 1):
        df = cost_sensitivity_analysis(results, weights_history,
                                       cost_levels=[bps], rf=rf)
        if df.empty:
            return np.nan
        sa = df[df["Method"] == method_a][f"cost_{bps}bps"]
        sb = df[df["Method"] == method_b][f"cost_{bps}bps"]
        if sa.empty or sb.empty:
            return np.nan
        if sa.values[0] <= sb.values[0]:
            return bps
    return max_bps


# ---------------------------------------------------------------------------
# Computational benchmarks
# ---------------------------------------------------------------------------

def spa_test(results, benchmark="EW", n_boot=1000, rf=0.03):
    """Hansen's Superior Predictive Ability (SPA) test.

    Tests H0: no method is significantly better than the benchmark.
    Returns dict with SPA p-value and best method.
    """
    methods = sorted(results["method"].unique())
    if benchmark not in methods:
        return {"spa_pvalue": np.nan, "best_method": None}
    others = [m for m in methods if m != benchmark]

    bench_r = results[results["method"] == benchmark].set_index("date")["return"]
    d_matrix = []
    for m in others:
        m_r = results[results["method"] == m].set_index("date")["return"]
        merged = pd.concat([m_r.rename("m"), bench_r.rename("b")], axis=1).dropna()
        d = merged["m"].values - merged["b"].values
        d_matrix.append(d)

    # Align lengths
    min_len = min(len(d) for d in d_matrix)
    d_matrix = np.column_stack([d[:min_len] for d in d_matrix])
    T = d_matrix.shape[0]
    d_means = d_matrix.mean(axis=0)

    # Stationary bootstrap
    np.random.seed(42)
    max_t_star = np.zeros(n_boot)
    for b in range(n_boot):
        idx = np.random.choice(T, T, replace=True)
        boot_means = d_matrix[idx].mean(axis=0) - d_means
        boot_stds = d_matrix[idx].std(axis=0) + 1e-12
        max_t_star[b] = np.max(boot_means / boot_stds * np.sqrt(T))

    # Observed test statistic
    obs_stds = d_matrix.std(axis=0) + 1e-12
    obs_t = np.max(d_means / obs_stds * np.sqrt(T))
    spa_p = np.mean(max_t_star >= obs_t)
    best_idx = np.argmax(d_means)

    return {
        "spa_pvalue": spa_p,
        "best_method": others[best_idx],
        "best_excess_sharpe": compute_sharpe(
            results[results["method"] == others[best_idx]]["return"].values, rf
        ) - compute_sharpe(bench_r.values, rf),
    }


def model_confidence_set(results, alpha=0.05, n_boot=1000, rf=0.03):
    """Model Confidence Set (Hansen, Lunde, Nason 2011).

    Returns the set of models that cannot be distinguished from the best
    at significance level alpha.
    """
    methods = sorted(results["method"].unique())
    if len(methods) < 2:
        return methods

    # Compute loss matrix (negative Sharpe as loss)
    sharpes = {}
    for m in methods:
        r = results[results["method"] == m]["return"].dropna().values
        sharpes[m] = compute_sharpe(r, rf)

    # Start with all models
    mcs = list(methods)
    eliminated = True

    while eliminated and len(mcs) > 1:
        eliminated = False
        worst_method = None
        worst_p = 1.0

        for m in mcs:
            others_sharpe = np.mean([sharpes[o] for o in mcs if o != m])
            d = sharpes[m] - others_sharpe
            if d < 0:
                # Bootstrap test for elimination
                m_r = results[results["method"] == m]["return"].dropna().values
                all_r = {o: results[results["method"] == o]["return"].dropna().values
                         for o in mcs if o != m}
                np.random.seed(42)
                boot_count = 0
                n = min(len(m_r), min(len(v) for v in all_r.values()))
                for _ in range(n_boot):
                    idx = np.random.choice(n, n, replace=True)
                    boot_m = compute_sharpe(m_r[idx], rf)
                    boot_others = np.mean([compute_sharpe(v[idx], rf) for v in all_r.values()])
                    if boot_m - boot_others >= 0:
                        boot_count += 1
                p_val = boot_count / n_boot
                if p_val < worst_p:
                    worst_p = p_val
                    worst_method = m

        if worst_method is not None and worst_p < alpha:
            mcs.remove(worst_method)
            eliminated = True

    return mcs


def benchmark_efficiency(models_dict, n_assets=5, feature_dim=48, n_runs=100):
    """Benchmark inference time and parameter count for each model.

    Args:
        models_dict: dict mapping method name to (model, is_torch) tuple
        n_assets: number of assets for synthetic input
        feature_dim: feature dimension for synthetic input
        n_runs: number of forward passes to time
    Returns:
        DataFrame with [Method, Params, Inference_ms]
    """
    import torch

    rows = []

    for name, (model, is_torch) in models_dict.items():
        if is_torch:
            # Place synthetic inputs on same device as model
            dev = next(model.parameters()).device
            x_t = torch.randn(feature_dim, device=dev)
            S_t = torch.eye(n_assets, device=dev) * 0.04
            n_params = sum(p.numel() for p in model.parameters())
            model.eval()
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    model(x_t, S_t)
            # Timed runs
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_runs):
                    model(x_t, S_t)
            elapsed = (time.perf_counter() - t0) / n_runs * 1000  # ms
        else:
            n_params = 0
            t0 = time.perf_counter()
            for _ in range(n_runs):
                model(np.zeros(n_assets), np.eye(n_assets) * 0.04)
            elapsed = (time.perf_counter() - t0) / n_runs * 1000

        rows.append({"Method": name, "Params": n_params, "Inference_ms": elapsed})
    return pd.DataFrame(rows)
