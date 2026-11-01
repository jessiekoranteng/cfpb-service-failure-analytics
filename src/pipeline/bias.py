"""Stage 6b: Bias Detection Suite for representation bias in the data (before any model exists).

Five automated checks. Severity: BLOCK stops publication, WARN is reported and carried into the Module 4 fairness plan.
  1. Coverage: every audit group must have enough records to be audited (BLOCK below MIN_GROUP_N_BLOCK, WARN below MIN_GROUP_N_WARN).
     BLOCK applies to protected-tag and region groups only; small channel/product groups are capped at WARN.
  2. Representation drift: a group's yearly share must stay within SHARE_DRIFT_MAX of its overall share.
  3. Differential missingness: the Tags-recorded rate must not differ by more than MISSINGNESS_GAP_MAX across regions or channels.
  4. Outcome disparity (Fairlearn MetricFrame): relief rate and untimely rate per group, parity difference and ratio (diagnostic).
  5. Concentration: share of the five largest companies (a dataset dominated by a few firms under-represents the rest).
"""
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, count, demographic_parity_difference, demographic_parity_ratio, selection_rate

from src.pipeline import config as C

GROUPS = ["tag_group", "region", "Submitted via", "Product"]
LABELS = ["monetary_relief", "untimely"]
# Groups whose under-representation can block publication (people/place). Channel and product are context
# dimensions: a tiny channel (e.g. 7 e-mail complaints) is reported as WARN but cannot block the run.
BLOCKING_GROUPS = {"tag_group", "region"}


def _coverage(d, min_block, min_warn):
    rows, level = [], "PASS"
    for g in GROUPS:
        for val, n in d[g].value_counts().items():
            sev = "BLOCK" if n < min_block else "WARN" if n < min_warn else "PASS"
            if sev == "BLOCK" and g not in BLOCKING_GROUPS:
                sev = "WARN"
            if g == "tag_group" and val == "None recorded":
                sev = "PASS"
            rows.append({"group": g, "value": str(val), "n": int(n), "share": round(n / len(d), 4), "severity": sev})
    sev = {r["severity"] for r in rows}
    level = "BLOCK" if "BLOCK" in sev else "WARN" if "WARN" in sev else "PASS"
    return {"status": level, "groups": rows}


def _drift(d):
    rows = []
    for g in GROUPS:
        overall = d[g].value_counts(normalize=True)
        by_year = d.groupby("year")[g].value_counts(normalize=True).unstack(fill_value=0)
        gap = (by_year - overall.reindex(by_year.columns).values).abs().max()
        for val, v in gap.items():
            rows.append({"group": g, "value": str(val), "max_abs_share_change": round(float(v), 4),
                         "severity": "WARN" if v > C.SHARE_DRIFT_MAX else "PASS"})
    return {"status": "WARN" if any(r["severity"] == "WARN" for r in rows) else "PASS", "groups": rows}


def _missingness(d):
    rows = []
    for g in ("region", "Submitted via", "year"):
        rate = d.groupby(g)["tag_recorded"].mean()
        gap = float(rate.max() - rate.min())
        rows.append({"by": g, "min_rate": round(float(rate.min()), 4), "max_rate": round(float(rate.max()), 4),
                     "gap": round(gap, 4), "severity": "WARN" if gap > C.MISSINGNESS_GAP_MAX else "PASS"})
    return {"status": "WARN" if any(r["severity"] == "WARN" for r in rows) else "PASS", "checks": rows}


def _outcomes(d):
    out = {}
    closed = d[d["monetary_relief"].notna()]
    for label, frame in (("monetary_relief", closed), ("untimely", d)):
        y = frame[label].astype(int)
        res = {}
        for g in ("tag_group", "region", "Submitted via"):
            mf = MetricFrame(metrics={"rate": selection_rate, "n": count}, y_true=y, y_pred=y, sensitive_features=frame[g])
            diff = float(demographic_parity_difference(y, y, sensitive_features=frame[g]))
            ratio = float(demographic_parity_ratio(y, y, sensitive_features=frame[g]))
            by = mf.by_group.reset_index()
            by.columns = ["value", "rate", "n"]
            res[g] = {"difference": round(diff, 4), "ratio": round(ratio, 4) if np.isfinite(ratio) else None,
                      "severity": "WARN" if (ratio < C.PARITY_RATIO_MIN) else "PASS",
                      "by_group": [{"value": str(r.value), "rate": round(float(r.rate), 4), "n": int(r.n)} for r in by.itertuples()]}
        out[label] = res
    flag = any(v["severity"] == "WARN" for lab in out.values() for v in lab.values())
    return {"status": "WARN" if flag else "PASS", "note": "diagnostic only; equalized odds is evaluated on model predictions in Module 4",
            "labels": out}


def _concentration(d):
    share = d["company_key"].value_counts(normalize=True)
    top5 = float(share.head(5).sum())
    hhi = float((share ** 2).sum())
    return {"status": "WARN" if top5 > C.TOP5_COMPANY_SHARE_WARN else "PASS", "top5_share": round(top5, 4),
            "hhi": round(hhi, 4), "n_companies": len(share)}


def bias_suite(df: pd.DataFrame, min_block: int = C.MIN_GROUP_N_BLOCK, min_warn: int = C.MIN_GROUP_N_WARN) -> dict:
    """Run all five checks on the curated table (needs tag_group, region, year, tag_recorded, labels, company_key)."""
    checks = {"coverage": _coverage(df, min_block, min_warn), "representation_drift": _drift(df), "differential_missingness": _missingness(df),
              "outcome_disparity": _outcomes(df), "concentration": _concentration(df)}
    statuses = [v["status"] for v in checks.values()]
    overall = "BLOCK" if "BLOCK" in statuses else "WARN" if "WARN" in statuses else "PASS"
    return {"overall": overall, "n_records": len(df), "checks": checks}


def assert_publishable(report: dict) -> None:
    if report["overall"] == "BLOCK":
        raise ValueError("bias suite BLOCK: an audit group has too few records to audit")
