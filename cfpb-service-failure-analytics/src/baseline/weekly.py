"""Comparable-period baseline and surge flags (goal G3).

Weekly complaint counts per group are compared with the mean and standard deviation of the previous
`window` weeks (the current week is never part of its own baseline). A week is flagged when
z >= `z_thresh` and the count exceeds `min_count`.
"""
import numpy as np
import pandas as pd


def weekly_counts(df, by=("Company", "Issue"), date_col="Date received"):
    d = df.copy()
    d["week_start"] = pd.to_datetime(d[date_col]).dt.to_period("W-SUN").dt.start_time
    g = d.groupby([*by, "week_start"]).size().rename("n").reset_index()
    return g


def surge_flags(counts, by=("Company", "Issue"), window=26, z_thresh=3.0, min_count=10):
    out = []
    for key, grp in counts.groupby(list(by)):
        grp = grp.set_index("week_start")["n"]
        full = grp.reindex(pd.date_range(grp.index.min(), grp.index.max(), freq="W-MON"), fill_value=0)
        prior_mean = full.shift(1).rolling(window, min_periods=8).mean()
        prior_std = full.shift(1).rolling(window, min_periods=8).std(ddof=1)
        z = (full - prior_mean) / prior_std.replace(0, np.nan)
        flag = (z >= z_thresh) & (full >= min_count)
        res = pd.DataFrame({"week_start": full.index, "n": full.values, "expected": prior_mean.values, "z": z.values, "surge": flag.values})
        for col, val in zip(by, key if isinstance(key, tuple) else (key,)):
            res[col] = val
        out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
