"""Build the aggregated data file the dashboard reads: app/assets/dashboard_data.json.

The deployed dashboard never sees complaint-level data. It gets (a) counts and rates per group, (b) score histograms per
group in fixed 0.0025 bins so a reviewer can move the review threshold and watch the fairness metrics change, (c) the
drop-down vocabulary, and (d) four worked example complaints. Everything is computed here from the Module 4 outputs and
the test window (2025-10 .. 2026-06), which the model has scored exactly once.

    python -m src.dashboard.build_assets
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

from src.modeling import config as C
from src.modeling import data as D
from src.modeling.charter import deployed_flags
from src.modeling.scoring import PLAIN

R = C.REPORTS
OUT = C.ROOT / "app" / "assets" / "dashboard_data.json"
BIN = 0.0025
NBINS = round(1 / BIN)
DIMS = {"region": "Region", "tag_older": "Older American", "tag_servicemember": "Servicemember", "tag_group": "Tag group"}


def _j(name):
    return json.loads((R / name).read_text())


def hist_by_group(g: pd.Series, y: np.ndarray, s: np.ndarray) -> dict:
    """{group: {"neg": [counts per bin], "pos": [...]}}: exact when a threshold sits on a bin edge."""
    b = np.minimum((s / BIN + 1e-9).astype(int), NBINS - 1)
    out = {}
    for name in sorted(g.astype(str).unique()):
        m = (g.astype(str) == name).to_numpy()
        out[name] = {"neg": np.bincount(b[m & (y == 0)], minlength=NBINS).tolist(), "pos": np.bincount(b[m & (y == 1)], minlength=NBINS).tolist()}
    return out


def confusion_by_group(g, y, f) -> dict:
    out = {}
    for name in sorted(g.astype(str).unique()):
        m = (g.astype(str) == name).to_numpy()
        yy, ff = y[m], f[m]
        out[name] = {"tn": int(((yy == 0) & (ff == 0)).sum()), "fp": int(((yy == 0) & (ff == 1)).sum()),
                     "fn": int(((yy == 1) & (ff == 0)).sum()), "tp": int(((yy == 1) & (ff == 1)).sum())}
    return out


def gains_table(y, s, n=10) -> list:
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    edges = np.linspace(0, len(y), n + 1).astype(int)
    rows, cum = [], 0
    for i in range(n):
        seg = ys[edges[i]:edges[i + 1]]
        cum += int(seg.sum())
        rows.append({"decile": i + 1, "complaints": len(seg), "relief_rate": float(seg.mean()), "cumulative_recall": cum / int(y.sum())})
    return rows


def calibration_table(y, s, n=10) -> list:
    q = np.quantile(s, np.linspace(0, 1, n + 1))
    idx = np.clip(np.searchsorted(q, s, side="right") - 1, 0, n - 1)
    return [{"bin": i + 1, "mean_score": float(s[idx == i].mean()), "observed_rate": float(y[idx == i].mean()), "n": int((idx == i).sum())} for i in range(n)]


def vocabulary(d: pd.DataFrame, bundle) -> dict:
    cats = bundle["base"]["coder"].categories_
    known = {k: set(map(str, v)) for k, v in cats.items()}
    d = d[d["Product"].isin(known["Product"])]
    sub_product = {p: sorted(x[x.isin(known["Sub-product"])].unique()) for p, x in d.groupby("Product")["Sub-product"]}
    issue = {p: sorted(x[x.isin(known["Issue"])].unique()) for p, x in d.groupby("Product")["Issue"]}
    sub_issue = {f"{p}||{i}": sorted(x[x.isin(known["Sub-issue"])].unique()) for (p, i), x in d.groupby(["Product", "Issue"])["Sub-issue"]}
    te = D.split(d, C.PRIMARY)["test"]
    vol = te.groupby("company_key")["company_volume_lag4"].median()
    counts = d["company_key"].value_counts()
    companies = [{"name": k, "complaints_since_2020": int(counts[k]), "typical_volume_4wk": None if pd.isna(vol.get(k, np.nan)) else float(vol[k])}
                 for k in sorted(known["company_key"]) if k in counts.index]
    companies.sort(key=lambda c: -c["complaints_since_2020"])
    return {"sub_product": sub_product, "issue": issue, "sub_issue": sub_issue, "companies": companies,
            "channel": sorted(known["Submitted via"]), "region": sorted(known["region"])}


def monthly_series(d: pd.DataFrame) -> list:
    m = d.assign(ym=d["week_start"].dt.to_period("M").astype(str)).groupby("ym").agg(
        complaints=("complaint_key", "size"), relief_rate=("monetary_relief", "mean"), untimely_rate=("untimely", "mean")).reset_index()
    return [{"month": r.ym, "complaints": int(r.complaints), "relief_rate": None if pd.isna(r.relief_rate) else float(r.relief_rate),
             "untimely_rate": float(r.untimely_rate)} for r in m.itertuples()]


def examples(te, relief) -> list:
    """Four worked complaints: A, B, C from the Module 4 SHAP notebook plus D (closest unflagged complaint below the line).
    Row numbers in local_cases.json index the notebook's 6,000-complaint test sample (seed 42), so the sample is rebuilt here
    the same way and every score is re-checked against the number the notebook reported."""
    rng = np.random.default_rng(C.SEED)
    S = te.reset_index(drop=True).iloc[rng.choice(len(te), 6000, replace=False)].reset_index(drop=True)
    sc = relief["calibrated"].predict_proba(S[C.RAW_INPUT])[:, 1]
    thr = relief["review_threshold"]
    cases = {c["case"][0]: c for c in _j("local_cases.json")}
    cf = {c["case"][0]: c for c in _j("counterfactuals.json")}
    rows = {k: int(cases[k]["row"]) for k in "ABC"}
    for k in "ABC":
        assert abs(sc[rows[k]] - cases[k]["score"]) < 6e-4, f"example {k} does not reproduce the Module 4 score"
    # same expression as the notebook (many complaints tie at the same isotonic step, so the tie-break must match)
    near = S.assign(score=sc)[sc < thr].assign(gap=lambda d: thr - d["score"]).sort_values("gap")
    rows["D"] = int(near[near["company_key"] != "BLOCK INC"].index[0])
    assert abs(sc[rows["D"]] - cf["D"]["score"]) < 6e-4 and S.iloc[rows["D"]]["company_key"] == cf["D"]["company"]
    titles = {"A": "Flagged, and relief followed", "B": "Relief followed; just under the single 17.5% line", "C": "Very low risk", "D": "No relief; just under the single 17.5% line"}
    short = {"A": "Clear flag", "B": "On the line", "C": "Very low risk", "D": "On the line"}
    out = []
    for k in "ABCD":
        r, i = S.iloc[rows[k]], rows[k]
        cut = relief["region_thresholds"].get(r["region"], thr)
        out.append({"id": k, "title": titles[k], "short": short[k], "inputs": {
            "product": r["Product"], "sub_product": r["Sub-product"], "issue": r["Issue"], "sub_issue": r["Sub-issue"], "channel": r["Submitted via"],
            "region": r["region"], "company": r["company_key"], "company_volume_lag4": None if pd.isna(r["company_volume_lag4"]) else float(r["company_volume_lag4"]),
            "month": int(r["month"])}, "score": round(float(sc[i]), 4), "flagged_global": bool(sc[i] >= thr), "region_cut": round(float(cut), 4),
            "flagged_regional": bool(sc[i] >= cut), "relief_happened": int(r[C.PRIMARY]), "what_if": cf.get(k, {}).get("changes", []),
            "top_contributions": cases.get(k, {}).get("top_contributions", [])[:3]})
    return out


def goals(facts, ut, untimely) -> list:
    """Module 1 goals G1-G5 against what Module 4 measured. Targets are quoted from the Vision Document (Table 1)."""
    y = ut[C.SECONDARY].to_numpy()
    sc = untimely["calibrated"].predict_proba(ut[C.RAW_INPUT])[:, 1]
    top = np.argsort(-sc, kind="stable")[: round(0.10 * len(y))]
    cap10 = float(y[top].sum() / y.sum())
    dep, perf = facts["mitigation_equal_opportunity"], facts["champion_calibrated_test"][C.PRIMARY]
    f2 = facts["charter"]["deployed_region_thresholds"]["f2"]
    return [
        {"id": "G1", "goal": "Predict costly outcomes (monetary relief)", "target": "ROC-AUC >= 0.75, and recall >= 60% at precision >= 25%",
         "observed": f"ROC-AUC {perf['roc_auc']:.2f}; recall {dep['recall']:.0%} at precision {dep['precision']:.0%}", "status": "Partly met",
         "note": "Ranking target met; recall target not met at the deployed review lines."},
        {"id": "G2", "goal": "Predict missed deadlines", "target": "Top 10% of scores capture >= 40% of untimely responses",
         "observed": f"Top 10% capture {cap10:.0%} (test)", "status": "Met, advisory", "note": "Late-response model is unstable over time (validation ROC-AUC 0.759, test 0.892)."},
        {"id": "G3", "goal": "Detect service-failure surges", "target": "Weekly volume >= 3 SD above baseline flagged within 7 days",
         "observed": "Built in Module 3 (baseline and surge flags)", "status": "Not re-assessed", "note": "Outside the Module 4 model; shown here as context only."},
        {"id": "G4", "goal": "Fair prioritisation", "target": "Equal-opportunity gap <= 0.05 for Older American and Servicemember",
         "observed": f"Older American {max(f2['older_american']['tpr_gap']['gap'], f2['older_american']['fpr_gap']['gap']):.3f} (interval reaches "
                     f"{max(f2['older_american']['tpr_gap']['ci95'][1], f2['older_american']['fpr_gap']['ci95'][1]):.3f}); Servicemember "
                     f"{max(f2['servicemember']['tpr_gap']['gap'], f2['servicemember']['fpr_gap']['gap']):.3f}", "status": "Partly met",
         "note": "Four-group comparison fails (0.061); proxy test F3 not met."},
        {"id": "G5", "goal": "Better triage policy", "target": "Risk-ranked queue cuts untimely responses >= 15% at fixed staffing",
         "observed": "Triage simulation not yet run", "status": "Not assessed", "note": "No claim is made about operational savings."},
    ]


def main():
    a, g = D.load()
    d = D.prepare(a)
    relief = joblib.load(C.MODELS / f"risk_{C.PRIMARY}.joblib")
    untimely = joblib.load(C.MODELS / f"risk_{C.SECONDARY}.joblib")
    joblib.load(C.MODELS / f"risk_{C.SECONDARY}.joblib")
    parts = D.split(d, C.PRIMARY)
    te = D.with_groups(parts["test"], g)
    y = te[C.PRIMARY].to_numpy()
    s = relief["calibrated"].predict_proba(te[C.RAW_INPUT])[:, 1]
    glob = (s >= relief["review_threshold"]).astype(int)
    dep = deployed_flags(relief, te, s)
    groups = {"region": te["region"].astype(str), "tag_older": te["tag_older"].map({1: "Older American", 0: "Not tagged"}),
              "tag_servicemember": te["tag_servicemember"].map({1: "Servicemember", 0: "Not tagged"}), "tag_group": te["tag_group"].astype(str)}
    facts = _j("facts.json")
    ut = D.split(d, C.SECONDARY)["test"]
    perf_u = facts["champion_calibrated_test"][C.SECONDARY]
    data = {
        "meta": {"records_in_scope": len(d), "test_window": ["2025-10-01", C.WINDOW_END], "n_test": len(y), "relief_model": relief["algorithm"],
                 "trained_at": relief["trained_at"], "review_capacity": C.REVIEW_CAPACITY, "global_threshold": float(relief["review_threshold"]),
                 "region_thresholds": {k: float(v) for k, v in relief["region_thresholds"].items()}, "bin_width": BIN,
                 "source": "Consumer Financial Protection Bureau, Consumer Complaint Database (public)"},
        "performance": {"relief": facts["champion_calibrated_test"][C.PRIMARY], "untimely": perf_u,
                        "deployed": {"flag_share": facts["mitigation_equal_opportunity"]["flag_share"], "precision": facts["mitigation_equal_opportunity"]["precision"],
                                     "recall": facts["mitigation_equal_opportunity"]["recall"]},
                        "capacity": facts["capacity"][C.PRIMARY], "vs_logreg": facts["champion_vs_logreg"]["pr_auc"],
                        "calibration": facts["calibration"][C.PRIMARY], "n_untimely_test": len(ut)},
        "gains": gains_table(y, s), "calibration_bins": calibration_table(y, s),
        "shap_global": [{"label": r["label"], "share": float(r["share"])} for r in facts["shap_global"]],
        "quarterly": facts["sensitivity"]["quarterly"], "perturbations": facts["sensitivity"]["perturbations"],
        "monthly": monthly_series(d[d["week_start"] <= pd.Timestamp(C.WINDOW_END)]),
        "criteria": facts["criteria"],
        "fairness": {
            "dims": DIMS,
            "hist": {k: hist_by_group(v, y, s) for k, v in groups.items()},
            "deployed_counts": {k: confusion_by_group(v, y, dep) for k, v in groups.items()},
            "global_counts": {k: confusion_by_group(v, y, glob) for k, v in groups.items()},
            "region_before": facts["region_before"], "region_after": facts["mitigation_equal_opportunity"],
            "threshold_optimizer": {k: facts["mitigation_threshold_optimizer"][k] for k in ("method", "before", "after", "balanced_accuracy_before", "balanced_accuracy_after")},
            "charter": {"limit": facts["charter"]["limit"], "f2": facts["charter"]["deployed_region_thresholds"]["f2"], "f3": facts["charter"]["f3_proxy"],
                        "f3_drivers": facts["proxy_drivers"]},
            "tags_global": facts["fairness_tags"],
        },
        "goals": goals(facts, ut, untimely),
        "vocab": vocabulary(d, relief),
        "examples": examples(te, relief),
        "plain_labels": PLAIN,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":"), default=float))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
    print("examples:", [(e["id"], e["score"], e["flagged_global"], e["flagged_regional"], e["inputs"]["region"]) for e in data["examples"]])


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    main()
