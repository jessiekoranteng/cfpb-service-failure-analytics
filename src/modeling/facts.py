"""Collect every number quoted in the Module 4 documents into one file, computed from the saved run outputs.

Nothing here is typed by hand: the Model Card, the Fairness Metrics Report and the Model Validation Report are all
rendered from `reports/m4/facts.json`, and each pre-registered criterion is evaluated here (pass / fail), including the
ones the model misses.

Run:  python -m src.modeling.facts
"""
import json
import os

import mlflow
import pandas as pd

from src.modeling import config as C

R = C.REPORTS
UNT_ = C.SECONDARY


def _j(name):
    return json.loads((R / name).read_text())


def criteria_table(m, a, cmp_, cap, ch):
    """Evaluate the pre-registered criteria (docs/module4/success_criteria.md) on the test window."""
    crit = C.CRITERIA[C.PRIMARY]
    perf = a["monetary_relief"]["performance"]
    rows = [
        ("Relief ROC-AUC", f">= {crit['roc_auc_min']:.2f}", perf["roc_auc"], perf["roc_auc"] >= crit["roc_auc_min"]),
        ("Relief PR-AUC lift over prevalence", f">= {crit['pr_auc_lift_min']:.1f}x", perf["pr_auc_lift"], perf["pr_auc_lift"] >= crit["pr_auc_lift_min"]),
        ("Relief recall when exactly the top 20% are flagged", f">= {crit['recall_at_capacity_min']:.2f}", cap[C.PRIMARY]["recall_top20"], cap[C.PRIMARY]["recall_top20"] >= crit["recall_at_capacity_min"]),
        ("Relief beats majority-class baseline (PR-AUC)", "greater", perf["pr_auc"] - m["monetary_relief"]["models"]["majority"]["test"]["pr_auc"], perf["pr_auc"] > m["monetary_relief"]["models"]["majority"]["test"]["pr_auc"]),
        ("Relief beats logistic-regression baseline (PR-AUC, paired bootstrap)", "difference > 0", cmp_["pr_auc"]["difference"], cmp_["pr_auc"]["difference"] > 0),
    ]
    u = a["untimely"]["performance"]
    rows.append(("Untimely PR-AUC lift over prevalence", f">= {C.CRITERIA[C.SECONDARY]['pr_auc_lift_min']:.1f}x", u["pr_auc_lift"], u["pr_auc_lift"] >= C.CRITERIA[C.SECONDARY]["pr_auc_lift_min"]))
    f = C.CRITERIA["fairness"]
    for g, label in (("older_american", "Older American"), ("servicemember", "Servicemember")):
        fg = a["monetary_relief"]["fairness"][g]
        rows += [
            (f"{label}: disparate impact ratio", f">= {f['disparate_impact_min']:.2f}", fg["disparate_impact_ratio"], fg["disparate_impact_ratio"] >= f["disparate_impact_min"]),
            (f"{label}: equalized-odds difference", f"<= {f['equalized_odds_diff_max']:.2f}", fg["equalized_odds_difference"], fg["equalized_odds_difference"] <= f["equalized_odds_diff_max"]),
            (f"{label}: demographic-parity difference", f"<= {f['demographic_parity_diff_max']:.2f}", fg["demographic_parity_difference"], fg["demographic_parity_difference"] <= f["demographic_parity_diff_max"]),
        ]
    rows = [(c, t, o, ok, "") for c, t, o, ok in rows]
    dep = ch["deployed_region_thresholds"]["f2"]
    for key, label in (("older_american", "Older American"), ("servicemember", "Servicemember"), ("four_group", "all four tag groups (incl. 'Both')")):
        d = dep[key]
        worst = max(("tpr_gap", "fpr_gap"), key=lambda m_: d[m_]["gap"])
        upper = max(d["tpr_gap"]["ci95"][1], d["fpr_gap"]["ci95"][1])
        rows.append((f"Charter F2: TPR and FPR gaps, {label} (deployed)", "<= 0.05", d[worst]["gap"], d["holds"],
                     f"95% CI upper bound {upper:.3f}" + ("" if d["holds_with_confidence"] else "; limit not confirmed")))
    for name, label in (("older_american", "Older American"), ("servicemember", "Servicemember")):
        p = ch["f3_proxy"][name]
        rows.append((f"Charter F3: {label} tag recoverable from inputs (AUC)", "95% CI includes 0.50", p["auc_recover_tag_from_features"],
                     p["ci95"][0] <= 0.5 <= p["ci95"][1], f"95% CI {p['ci95'][0]:.3f}-{p['ci95'][1]:.3f}"))
    return [{"criterion": c, "target": t, "observed": float(o), "met": bool(ok), "note": n} for c, t, o, ok, n in rows]


def mlflow_counts():
    mlflow.set_tracking_uri(C.MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(C.EXPERIMENT)
    runs = client.search_runs([exp.experiment_id], max_results=1000)
    names = [r.data.tags.get("mlflow.runName", "") for r in runs]
    nested = [r for r in runs if "mlflow.parentRunId" in r.data.tags]
    reg = {}
    for rm in client.search_registered_models():
        reg[rm.name] = {"versions": len(client.search_model_versions(f"name='{rm.name}'")), "aliases": dict(rm.aliases)}
    return {"experiment": C.EXPERIMENT, "experiment_id": exp.experiment_id, "runs_total": len(runs), "tuning_trials_nested": len(nested),
            "run_names": sorted(set(names)), "registry": reg}


def build():
    m = {t: _j(f"metrics_{t}.json") for t in C.TARGETS}
    a = {t: _j(f"analysis_{t}.json") for t in C.TARGETS}
    cmp_ = _j(f"champion_vs_logreg_{C.PRIMARY}.json")
    shap = pd.read_csv(R / "shap_global_importance.csv")
    api = _j("api_live_demo.json")
    cap = _j("capacity_check.json")
    ch = _j("charter_checks.json")
    drivers = _j("proxy_drivers.json") if (R / "proxy_drivers.json").exists() else {}
    rel = a[C.PRIMARY]
    facts = {
        "split": {t: m[t]["split"] for t in C.TARGETS},
        "models": {t: {k: {"test": v["test"], "valid": v["valid"], "cv": v.get("cv", {}), "run_id": v.get("run_id")} for k, v in m[t]["models"].items()} for t in C.TARGETS},
        "champion": {t: m[t]["champion"] for t in C.TARGETS},
        "champion_calibrated_test": {t: a[t]["performance"] for t in C.TARGETS},
        "calibration": {t: a[t]["calibration"] for t in C.TARGETS},
        "relief_at_review_threshold": rel["performance_at_review_threshold"],
        "deployed_threshold": rel["deployed_threshold"],
        "test_flag_share": rel["test_flag_share"],
        "untimely_test_flag_share": a[UNT_]["test_flag_share"],
        "capacity": cap,
        "charter": ch,
        "proxy_drivers": drivers,
        "criteria": criteria_table(m, a, cmp_, cap, ch),
        "champion_vs_logreg": cmp_,
        "fairness_tags": rel["fairness"],
        "fairness_untimely_di": {g: a["untimely"]["fairness"][g]["disparate_impact_ratio"] for g in ("older_american", "servicemember")},
        "region_before": rel["fairness"]["region"],
        "tag_group_before": rel["fairness"]["tag_group"],
        "mitigation_threshold_optimizer": rel["mitigation"],
        "mitigation_equal_opportunity": rel["mitigation_equal_opportunity"],
        "sensitivity": rel["sensitivity"],
        "shap_global": shap.to_dict("records"),
        "api_demo": [{"status": x["status"], "request": x["request"], "response": x["response"]} for x in api],
        "mlflow": mlflow_counts(),
    }
    return facts


def main():
    f = build()
    (R / "facts.json").write_text(json.dumps(f, indent=2, default=str))
    unmet = [c for c in f["criteria"] if not c["met"]]
    print(f"criteria: {len(f['criteria']) - len(unmet)}/{len(f['criteria'])} met")
    for c in unmet:
        print("  NOT MET:", c["criterion"], round(c["observed"], 4), c.get("note", ""))
    print("mlflow:", {k: v for k, v in f["mlflow"].items() if k != "run_names"})


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    main()
