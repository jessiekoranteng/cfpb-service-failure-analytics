"""Data-contract gate for the scoped CFPB extract. Returns violations; never silently repairs data."""
import pandas as pd

REQUIRED = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue", "Company", "State",
    "Tags", "Submitted via", "Company response to consumer", "Timely response?", "Complaint ID",
]
PROHIBITED = {"ZIP code", "Company public response", "Date sent to company", "Consumer complaint narrative"}
ALLOWED_TIMELY = {"Yes", "No"}
ALLOWED_TAGS = {"Older American", "Servicemember", "Older American, Servicemember"}
ALLOWED_PRODUCTS = {"Checking or savings account", "Money transfer, virtual currency, or money service"}
NO_NULL = ["Date received", "Product", "Issue", "Company", "Submitted via", "Company response to consumer", "Timely response?", "Complaint ID"]


def check_contract(df):
    """Return a list of human-readable violations (empty list means the gate passes)."""
    v = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    banned = sorted(PROHIBITED.intersection(df.columns))
    if banned:
        v.append(f"prohibited columns present: {banned}")
    for c in NO_NULL:
        n = int(df[c].isna().sum())
        if n:
            v.append(f"{c}: {n} unexpected nulls")
    if df["Complaint ID"].duplicated().any():
        v.append(f"Complaint ID: {int(df['Complaint ID'].duplicated().sum())} duplicates")
    bad = set(df["Timely response?"].dropna().unique()) - ALLOWED_TIMELY
    if bad:
        v.append(f"Timely response? unexpected values: {sorted(bad)}")
    bad = set(df["Tags"].dropna().unique()) - ALLOWED_TAGS
    if bad:
        v.append(f"Tags unexpected values: {sorted(bad)}")
    bad = set(df["Product"].dropna().unique()) - ALLOWED_PRODUCTS
    if bad:
        v.append(f"Product out of scope: {sorted(bad)}")
    dates = pd.to_datetime(df["Date received"], errors="coerce")
    if dates.isna().any():
        v.append("Date received: unparseable values")
    elif dates.min() < pd.Timestamp("2020-01-01"):
        v.append("Date received earlier than scope start 2020-01-01")
    return v


def assert_contract(df):
    violations = check_contract(df)
    if violations:
        raise ValueError("data contract failed: " + "; ".join(violations))
