"""Stage 5: integration. Joins the cleaned complaints with the state-to-region reference table and a company-level
weekly volume aggregate, and builds the dashboard feed (weekly company-issue counts with surge flags, goal G3)."""
import numpy as np
import pandas as pd

from src.baseline.weekly import surge_flags, weekly_counts

MODEL_FEATURES = ["Product", "Sub-product", "Issue", "Sub-issue", "Submitted via", "region", "company_key",
                  "company_volume_lag4", "week_start", "year"]
LABELS = ["monetary_relief", "untimely"]
AUDIT_ONLY = ["tag_recorded", "tag_older", "tag_servicemember", "tag_group"]


def load_state_region(path) -> pd.DataFrame:
    ref = pd.read_csv(path, dtype=str)
    if ref["state"].duplicated().any():
        raise ValueError("state_region reference has duplicate state codes")
    return ref


def add_region(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    m = df.merge(ref, how="left", left_on="State", right_on="state", validate="m:1")
    m["region"] = m["region"].fillna("Other")
    return m.drop(columns=["state"])


def company_lag4(df: pd.DataFrame) -> pd.Series:
    """Complaints about the same company in the previous 4 weeks (the current week is never included)."""
    cw = df.groupby(["company_key", "week_start"]).size().rename("n").reset_index()
    weeks = pd.date_range(cw["week_start"].min(), cw["week_start"].max(), freq="7D")
    grid = pd.MultiIndex.from_product([cw["company_key"].unique(), weeks], names=["company_key", "week_start"])
    full = cw.set_index(["company_key", "week_start"])["n"].reindex(grid, fill_value=0).reset_index()
    full["company_volume_lag4"] = (
        full.groupby("company_key")["n"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).sum()).fillna(0)
    )
    key = full.set_index(["company_key", "week_start"])["company_volume_lag4"]
    idx = pd.MultiIndex.from_frame(df[["company_key", "week_start"]])
    return pd.Series(key.reindex(idx).to_numpy(), index=df.index).fillna(0).astype("float32")


def integrate(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    d = add_region(df, ref)
    d["company_volume_lag4"] = company_lag4(d)
    return d


def dashboard_feed(df: pd.DataFrame, min_total: int = 200, last_weeks: int = 104) -> pd.DataFrame:
    """Weekly company-issue counts with surge flags for pairs with at least `min_total` complaints."""
    counts = weekly_counts(df, by=("company_key", "Issue"), date_col="week_start")
    keep = counts.groupby(["company_key", "Issue"])["n"].transform("sum") >= min_total
    flags = surge_flags(counts[keep], by=("company_key", "Issue"))
    if flags.empty:
        return flags
    cutoff = flags["week_start"].max() - pd.Timedelta(weeks=last_weeks)
    out = flags[flags["week_start"] > cutoff].copy()
    out["expected"] = out["expected"].round(2)
    out["z"] = out["z"].replace([np.inf, -np.inf], np.nan).round(2)
    return out.reset_index(drop=True)


def split_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model-ready table (no audit-only fields, no source identifiers) and the restricted audit table."""
    analysis = df[["complaint_key", *MODEL_FEATURES, *LABELS]].copy()
    audit = df[["complaint_key", "year", "region", *AUDIT_ONLY, *LABELS]].copy()
    return analysis, audit
