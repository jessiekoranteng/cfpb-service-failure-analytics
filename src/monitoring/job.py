"""Stage 2 of monitoring (Evidently environment, pandas 2): drift, performance, fairness, alerts.

    python -m src.monitoring.job --scored data/processed/monitoring_scored.parquet \
        --out monitoring_app/assets/monitoring_data.json --html monitoring_app/assets/evidently_drift_report.html [--store explanations.jsonl]

Evidently produces the data-drift report (reference window vs replay window, plus prediction drift). Fairlearn produces
the group tables; the F2 gaps use the repository's own definition (group vs everyone else) with bootstrap intervals, and
the two are cross-checked in the output (`crosscheck`). Alert levels come from src/monitoring/thresholds.py.
"""
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.monitoring import metrics as M
from src.monitoring import thresholds as T

DRIFT_COLS_CAT = ["Product", "Issue", "Submitted via", "region", "company_key"]
DRIFT_COLS_NUM = ["company_volume_lag4", "relief_probability"]
SAMPLE = 20000


def monthly(ref: pd.DataFrame, cur: pd.DataFrame) -> list[dict]:
    rows = []
    for m, d in cur.groupby("month_received"):
        row = {"month": m, "n": len(d), "relief_rate": float(d["y"].mean()), "flag_share": float(d["flag_for_review"].mean())}
        if len(d) >= T.MIN_MONTH_ROWS and d["y"].nunique() == 2:
            base = d["y"].mean()
            row["roc_auc"] = float(roc_auc_score(d["y"], d["relief_probability"]))
            pr = float(average_precision_score(d["y"], d["relief_probability"]))
            row["pr_auc_lift"] = pr / base
            row["recall_at_flag"] = float(d.loc[d["y"] == 1, "flag_for_review"].mean())
            row["ece"] = M.ece(d["y"], d["relief_probability"])
        else:
            row.update(roc_auc=None, pr_auc_lift=None, recall_at_flag=None, ece=None)
        row["unseen_company_share"] = float((~d["company_seen"]).mean())
        row["psi_company"] = M.psi_categorical(ref["company_key"], d["company_key"])
        row["psi_score"] = M.psi_numeric(ref["relief_probability"], d["relief_probability"])
        row["region_selection_ratio"] = M.selection_ratio(d["flag_for_review"], d["region"])
        row["smallest_region_n"] = int(d["region"].value_counts().min())   # a ratio over <300 rows in one region is noisy
        row["levels"] = {k: T.level(k, row.get(k)) for k in T.RULES if k in row}
        rows.append(row)
    return rows


def alerts_from(rows: list[dict]) -> list[dict]:
    return [{"month": r["month"], "signal": k, "value": r[k], "level": lv, "rule": T.RULES[k][3],
             "noisy": bool(k == "region_selection_ratio" and r["smallest_region_n"] < 300)}
            for r in rows for k, lv in r["levels"].items() if lv in ("warn", "hard")]


def fairness_tags(cur: pd.DataFrame, n_boot: int) -> dict:
    """Charter F2 on the replay window and per quarter. Tag coverage is only ~15%, so 'rest' includes untagged complaints."""
    def one(d):
        out = {}
        for name, col in (("older_american", "tag_older"), ("servicemember", "tag_servicemember")):
            g = d[col].astype(bool).to_numpy()
            gaps = M.vs_rest_gaps(d["y"], d["flag_for_review"], g)
            ci = M.bootstrap_gaps(d["y"], d["flag_for_review"], g, n_boot=n_boot) if gaps["tpr_gap"] is not None else {"tpr_ci95": None, "fpr_ci95": None}
            lv_t, lv_f = T.level("f2_tpr_gap", gaps["tpr_gap"]), T.level("f2_fpr_gap", gaps["fpr_gap"])
            out[name] = {**gaps, **ci, "tpr_level": lv_t, "fpr_level": lv_f,
                         "confirmed_breach": bool(ci["tpr_ci95"] and ci["tpr_ci95"][0] > 0.05) or bool(ci["fpr_ci95"] and ci["fpr_ci95"][0] > 0.05)}
        return out
    q = cur["week_start"].dt.to_period("Q").astype(str)
    return {"replay_window": one(cur), "by_quarter": {k: one(d) for k, d in cur.groupby(q)}}


def fairness_cuts(cur: pd.DataFrame) -> dict:
    """Group tables by product, company size tier and region, using Fairlearn's MetricFrame; cells under MIN_CELL are suppressed."""
    from fairlearn.metrics import MetricFrame, count, false_positive_rate, selection_rate, true_positive_rate
    out = {}
    for label, col in (("product", "Product"), ("company_size_tier", "size_tier"), ("region", "region")):
        mf = MetricFrame(metrics={"flag_rate": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate, "n": count},
                         y_true=cur["y"].to_numpy(), y_pred=cur["flag_for_review"].astype(int).to_numpy(), sensitive_features=cur[col].astype(str).to_numpy())
        rows = []
        for grp, r in mf.by_group.iterrows():
            gr = M.group_rates(cur["y"], cur["flag_for_review"], (cur[col].astype(str) == grp).to_numpy())
            assert abs(gr["flag_rate"] - r["flag_rate"]) < 1e-9 and int(r["n"]) == gr["n"], "own and Fairlearn group tables disagree"
            rows.append({"group": grp, **gr, "suppressed": gr["tpr"] is None or gr["fpr"] is None})
        rows.sort(key=lambda x: -x["n"])
        tprs = [x["tpr"] for x in rows if x["tpr"] is not None]
        fprs = [x["fpr"] for x in rows if x["fpr"] is not None]
        out[label] = {"groups": rows, "tpr_spread": float(max(tprs) - min(tprs)) if tprs else None,
                      "fpr_spread": float(max(fprs) - min(fprs)) if fprs else None,
                      "selection_ratio": M.selection_ratio(cur["flag_for_review"], cur[col])}
    return out


def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, html: Path | None) -> dict:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset
    cols = DRIFT_COLS_CAT + DRIFT_COLS_NUM
    r, c = ref[cols].sample(min(SAMPLE, len(ref)), random_state=42), cur[cols].sample(min(SAMPLE, len(cur)), random_state=42)
    dd = DataDefinition(numerical_columns=DRIFT_COLS_NUM, categorical_columns=DRIFT_COLS_CAT)
    snap = Report([DataDriftPreset()]).run(Dataset.from_pandas(c, data_definition=dd), Dataset.from_pandas(r, data_definition=dd))
    if html:
        snap.save_html(str(html))
    cols_out, share = [], None
    for m in snap.dict()["metrics"]:
        cfg = m["config"]
        if cfg["type"].endswith("DriftedColumnsCount"):
            share = m["value"]
        elif cfg["type"].endswith("ValueDrift"):
            pval = "p_value" in cfg["method"].lower() or "p-value" in cfg["method"].lower()
            drifted = m["value"] < cfg["threshold"] if pval else m["value"] > cfg["threshold"]
            cols_out.append({"column": cfg["column"], "method": cfg["method"], "threshold": cfg["threshold"], "value": float(m["value"]), "drifted": bool(drifted)})
    return {"evidently_version": __import__("evidently").__version__, "sample_per_window": len(r), "drifted_columns": share, "columns": cols_out}


def live_summary(store: Path | None) -> dict:
    if not store or not Path(store).exists():
        return {"available": False}
    recs = [json.loads(x) for x in Path(store).read_text().splitlines() if x.strip()]
    if not recs:
        return {"available": False}
    return {"available": True, "n": len(recs), "first": recs[0]["ts"], "last": recs[-1]["ts"],
            "flag_share": float(np.mean([r["flag_for_review"] for r in recs])),
            "unseen_share": float(np.mean([bool(r["unseen_values"]) for r in recs]))}


def run(scored: Path, out: Path, html: Path | None, store: Path | None = None, n_boot: int = 300) -> dict:
    df = pd.read_parquet(scored)
    ref, cur = df[df["window"] == "reference"], df[df["window"] == "replay"]
    rows = monthly(ref, cur)
    res = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "basis": "replay: complaints from the test window (never used in training or calibration) scored month by month with the deployed model, as if received live",
        "reference": {"window": f"{ref['month_received'].min()} to {ref['month_received'].max()}", "n": len(ref), "relief_rate": float(ref["y"].mean())},
        "replay": {"window": f"{cur['month_received'].min()} to {cur['month_received'].max()}", "n": len(cur), "relief_rate": float(cur["y"].mean()),
                   "flag_share": float(cur["flag_for_review"].mean())},
        "rules": {k: {"direction": v[0], "warn": v[1], "hard": v[2], "source": v[3]} for k, v in T.RULES.items()},
        "monthly": rows,
        "alerts": alerts_from(rows),
        "fairness_tags": fairness_tags(cur, n_boot),
        "fairness_cuts": fairness_cuts(cur),
        "drift": evidently_report(ref, cur, html),
        "live_api": live_summary(store),
        "limits": ["Replay, not live traffic: no human review or analyst override is observed, so analyst override rate is not computed.",
                   "Tag coverage is about 15%; 'rest' for the tag gaps includes untagged complaints (same definition as Module 4).",
                   f"Subgroup cells with fewer than {T.MIN_CELL} positives or negatives are suppressed."],
    }
    res["crosscheck"] = {"replay_roc_auc": float(roc_auc_score(cur["y"], cur["relief_probability"])), "replay_ece": M.ece(cur["y"], cur["relief_probability"])}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--html")
    ap.add_argument("--store")
    ap.add_argument("--n-boot", type=int, default=300)
    a = ap.parse_args()
    r = run(Path(a.scored), Path(a.out), Path(a.html) if a.html else None, Path(a.store) if a.store else None, a.n_boot)
    print("alerts:", len(r["alerts"]), "| drifted columns:", r["drift"]["drifted_columns"])


if __name__ == "__main__":
    main()
