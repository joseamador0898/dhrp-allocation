import os

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")


def load_ff5_factors(start, end):
    """Load Fama-French 5 factors + Momentum from Ken French's data library.

    Returns DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF
    """
    import pandas_datareader.data as pdr

    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"FF5Mom_{start}_{end}.csv")
    if os.path.exists(cp):
        ff = pd.read_csv(cp, index_col=0, parse_dates=True)
        print(f"  FF5+Mom factors: {ff.shape[0]} days (cached)")
        return ff

    # FF5 daily factors
    try:
        ff5 = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start, end)[0] / 100
    except Exception:
        ff5 = pdr.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start, end)[0] / 100

    # Momentum factor
    try:
        mom = pdr.DataReader("F-F_Momentum_Factor_daily", "famafrench", start, end)[0] / 100
        mom.columns = ["Mom"]
    except Exception:
        mom = pd.DataFrame(index=ff5.index, columns=["Mom"], data=0.0)

    ff = pd.concat([ff5, mom], axis=1, join="inner")
    ff.to_csv(cp)
    print(f"  FF5+Mom factors: {ff.shape[0]} days")
    return ff


def factor_analysis(results, ff, methods=None):
    """Run Fama-French factor regressions for each method.

    Automatically uses FF5+Mom if available, falls back to FF3.
    """
    if methods is None:
        methods = sorted(results["method"].unique())

    # Determine available factors
    available = ff.columns.tolist()
    core_factors = ["Mkt-RF", "SMB", "HML"]
    extra_factors = [f for f in ["RMW", "CMA", "Mom"] if f in available]
    all_factors = core_factors + extra_factors
    factor_cols = [f for f in all_factors if f in available]
    n_factors = len(factor_cols)

    stats = []
    for m in methods:
        ser = results[results["method"] == m].set_index("date")["return"]
        ser.index = pd.to_datetime(ser.index)
        merged = pd.concat([ser.rename("ret"), ff], axis=1, join="inner").dropna()
        if len(merged) < 100:
            continue

        X = merged[factor_cols].values
        y = merged["ret"].values
        mod = OLS(y, add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

        row = {
            "Method": m,
            "Alpha_ann": mod.params[0] * 252,
            "Alpha_t": mod.tvalues[0],
            "Alpha_p": mod.pvalues[0],
            "R2_adj": mod.rsquared_adj,
        }
        for i, f in enumerate(factor_cols):
            row[f"Beta_{f}"] = mod.params[i + 1]
            row[f"t_{f}"] = mod.tvalues[i + 1]

        stats.append(row)
    return pd.DataFrame(stats)
