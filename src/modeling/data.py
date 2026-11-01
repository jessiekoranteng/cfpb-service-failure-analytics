"""Load the Module 3 analysis table, build features and make the time-aware split."""
import pandas as pd

from src.modeling import config as C


def load(processed=C.PROCESSED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model-ready table and the restricted audit table (protected flags), joined only by the pseudonymous key."""
    a = pd.read_parquet(processed / "analysis_table.parquet")
    g = pd.read_parquet(processed / "audit_groups.parquet")
    return a, g


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw model inputs: month from the received week, categoricals as clean strings. Derived features live in the
    pipeline (FeatureBuilder) so training and the API share exactly one implementation."""
    d = df.copy()
    d["month"] = d["week_start"].dt.month.astype("int8")
    d["company_volume_lag4"] = d["company_volume_lag4"].astype("float32")
    for c in C.CATEGORICAL:
        d[c] = d[c].astype("string").fillna("Unknown").astype(object)
    return d


def assert_no_leakage(cols) -> None:
    bad = set(cols) & C.FORBIDDEN
    if bad:
        raise ValueError(f"forbidden columns in feature set: {sorted(bad)}")


def split(df: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    """Train / valid / test by received week. Rows without a label (case still open) and rows after WINDOW_END are dropped."""
    d = df[df["week_start"] <= pd.Timestamp(C.WINDOW_END)]
    d = d[d[target].notna()].copy()
    d[target] = d[target].astype("int8")
    tr = d[d["week_start"] <= pd.Timestamp(C.TRAIN_END)]
    va = d[(d["week_start"] > pd.Timestamp(C.TRAIN_END)) & (d["week_start"] <= pd.Timestamp(C.VALID_END))]
    te = d[d["week_start"] > pd.Timestamp(C.VALID_END)]
    return {"train": tr.sort_values("week_start").reset_index(drop=True), "valid": va.reset_index(drop=True), "test": te.reset_index(drop=True)}


def with_groups(df: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Attach protected flags for AUDIT use only (never passed to the model)."""
    cols = ["complaint_key", "tag_older", "tag_servicemember", "tag_group"]
    return df.merge(groups[cols], on="complaint_key", how="left", validate="one_to_one")
