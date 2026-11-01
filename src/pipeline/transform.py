"""Stage 4: transformation. Derives labels, time keys and audit-only flags exactly as defined in the data dictionary."""
import pandas as pd

RELIEF = "Closed with monetary relief"
OPEN_STATUS = "In progress"


def company_key(s: pd.Series) -> pd.Series:
    """Grouping key that ignores case and trailing punctuation so 'ACME, INC.' and 'Acme, Inc' group together."""
    return s.str.upper().str.replace(r"[^A-Z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def transform(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    resp = d["Company response to consumer"]
    closed = resp != OPEN_STATUS
    d["monetary_relief"] = pd.array((resp == RELIEF).astype("int8").where(closed), dtype="Int8")
    d["untimely"] = (d["Timely response?"] == "No").astype("int8")
    d["week_start"] = d["Date received"].dt.to_period("W-SUN").dt.start_time
    d["year"] = d["Date received"].dt.year.astype("int16")
    d["company_key"] = company_key(d["Company"])
    tags = d["Tags"].fillna("")
    d["tag_recorded"] = d["Tags"].notna().astype("int8")
    d["tag_older"] = tags.str.contains("Older American").astype("int8")
    d["tag_servicemember"] = tags.str.contains("Servicemember").astype("int8")
    d["tag_group"] = "None recorded"
    d.loc[d["tag_older"] == 1, "tag_group"] = "Older American"
    d.loc[d["tag_servicemember"] == 1, "tag_group"] = "Servicemember"
    d.loc[(d["tag_older"] == 1) & (d["tag_servicemember"] == 1), "tag_group"] = "Both"
    return d
