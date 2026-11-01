"""Stage 3: cleaning. Documented, counted, reversible-by-rerun; nothing is repaired silently.

Rules (each one is counted in the returned report):
  1. Trim whitespace and turn empty strings into missing values.
  2. Parse `Date received`; drop rows with an unparseable date or a date after the snapshot date.
  3. Drop exact duplicate rows, then duplicate `Complaint ID` values (keep the first).
  4. Missing-data policy: categorical gaps in Sub-product, Issue, Sub-issue and State become 'Unknown'; Tags stay missing
     because absence means 'no tag recorded', not 'not an older American'; a `tag_recorded` flag is added later.
  5. State codes are upper-cased.
"""
import pandas as pd

FILL_UNKNOWN = ["Sub-product", "Issue", "Sub-issue", "State"]


def clean(df: pd.DataFrame, snapshot_date: str | None = None) -> tuple[pd.DataFrame, dict]:
    rep = {"rows_in": len(df)}
    d = df.copy()
    obj = d.select_dtypes(include=["object", "string"]).columns
    changed = 0
    for c in obj:
        s = d[c]
        stripped = s.str.strip()
        changed += int((s.notna() & (s != stripped)).sum())
        d[c] = stripped.replace("", pd.NA)
    rep["whitespace_values_trimmed"] = changed

    dates = pd.to_datetime(d["Date received"], format="%Y-%m-%d", errors="coerce")
    bad = dates.isna()
    if snapshot_date is not None:
        bad |= dates > pd.Timestamp(snapshot_date)
    rep["bad_or_future_dates_dropped"] = int(bad.sum())
    d = d[~bad].copy()
    d["Date received"] = dates[~bad]

    before = len(d)
    d = d.drop_duplicates()
    rep["exact_duplicates_dropped"] = before - len(d)
    before = len(d)
    d = d.drop_duplicates(subset="Complaint ID", keep="first")
    rep["id_duplicates_dropped"] = before - len(d)

    rep["nulls_filled_unknown"] = {}
    for c in FILL_UNKNOWN:
        n = int(d[c].isna().sum())
        rep["nulls_filled_unknown"][c] = n
        d[c] = d[c].fillna("Unknown")
    d["State"] = d["State"].str.upper()
    rep["tags_missing"] = int(d["Tags"].isna().sum())
    d = d.reset_index(drop=True)
    rep["rows_out"] = len(d)
    return d, rep
