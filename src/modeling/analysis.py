"""Test-window fairness audit, mitigation experiment and sensitivity analysis for the champion models.

    python -m src.modeling.analysis

Writes reports/m4/analysis_<target>.json and figures, and logs the headline numbers to the same MLflow experiment.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.calibration import calibration_curve
from sklearn.metrics import balanced_accuracy_score

from src.modeling import config as C
from src.modeling import data as D
from src.modeling import fairness as F
from src.modeling import metrics as M
from src.modeling import pipeline as P
from src.modeling import sensitivity as S

TEAL, AMB, INK = "#0E7C86", "#F2A541", "#12263A"


def _split(target):
    a, g = D.load()
    a = D.prepare(a)
    sp = D.split(a, target)
    return {k: D.with_groups(v, g) for k, v in sp.items()}


def ece(y, s, bins=10):
    f, m = calibration_curve(y, s, n_bins=bins, strategy="quantile")
    w = np.histogram(s, bins=np.quantile(s, np.linspace(0, 1, bins + 1)))[0] / len(s)
    return float(np.sum(np.abs(f - m) * w[: len(f)]))


def fairness_plot(rep_before, rep_after, path):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    names = ["Older American", "Servicemember"]
    keys = ["older_american", "servicemember"]
    for a_, metric, title, line in zip(ax, ["disparate_impact_ratio", "equalized_odds_difference", "demographic_parity_difference"],
                                      ["Disparate impact ratio (need ≥ 0.80)", "Equalized-odds difference (need ≤ 0.10)", "Demographic-parity difference (need ≤ 0.10)"], [0.8, 0.1, 0.1]):
        vals = [rep_before[k][metric] for k in keys]
        a_.bar(names, vals, color=TEAL)
        a_.axhline(line, color="#C0392B", ls="--")
        a_.set_title(title, fontsize=9)
        for i, v in enumerate(vals):
            a_.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def region_plot(tbl_before, tbl_after, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    reg = [r["region"] for r in tbl_before]
    x = np.arange(len(reg))
    for a_, key, title in zip(ax, ["tpr", "fpr"], ["True-positive rate by region", "False-positive rate by region"]):
        a_.bar(x - 0.2, [r[key] for r in tbl_before], 0.4, color=TEAL, label="before mitigation")
        a_.bar(x + 0.2, [r[key] for r in tbl_after], 0.4, color=AMB, label="after mitigation")
        a_.set_xticks(x, reg)
        a_.set_title(title, fontsize=10)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mitigate_region(bundle, sp, target):
    """Post-processing: ThresholdOptimizer with an equalized-odds constraint across regions, fitted on the validation window."""
    Xva, yva = sp["valid"][C.RAW_INPUT], sp["valid"][target].to_numpy()
    Xte = sp["test"][C.RAW_INPUT]
    to = ThresholdOptimizer(estimator=bundle["calibrated"], constraints="equalized_odds", objective="balanced_accuracy_score",
                            prefit=True, predict_method="predict_proba")
    to.fit(Xva, yva, sensitive_features=sp["valid"]["region"].astype(str))
    pred = np.asarray(to.predict(Xte, sensitive_features=sp["test"]["region"].astype(str), random_state=C.SEED)).astype(int)
    return pred


def equal_opportunity_thresholds(bundle, sp, target):
    """Per-region cut-offs chosen on the validation window so each region's recall equals the overall recall at the deployed
    threshold (equal opportunity). Needs only the region, which is already a model input."""
    cal, thr = bundle["calibrated"], bundle["review_threshold"]
    va, te = sp["valid"], sp["test"]
    sv = cal.predict_proba(va[C.RAW_INPUT])[:, 1]
    yv = va[target].to_numpy()
    tpr_target = float((sv[yv == 1] >= thr).mean())
    cuts = {}
    for r, idx in va.groupby(va["region"].astype(str)).groups.items():
        m = va.index.get_indexer(idx)
        pos = sv[m][yv[m] == 1]
        cuts[r] = float(np.quantile(pos, 1 - tpr_target)) if len(pos) >= 30 else float(thr)
    st = cal.predict_proba(te[C.RAW_INPUT])[:, 1]
    cut = te["region"].astype(str).map(cuts).to_numpy()
    return (st >= cut).astype(int), cuts, tpr_target


def run_target(target):
    sp = _split(target)
    bundle = joblib.load(C.MODELS / f"risk_{target}.joblib")
    cal, base = bundle["calibrated"], bundle["base"]
    te, va, tr = sp["test"], sp["valid"], sp["train"]
    Xte, yte = te[C.RAW_INPUT], te[target].to_numpy()
    s = cal.predict_proba(Xte)[:, 1]
    sva = cal.predict_proba(va[C.RAW_INPUT])[:, 1]
    thr = float(bundle["review_threshold"])
    pred = (s >= thr).astype(int)
    out = {"target": target, "algorithm": bundle["algorithm"], "deployed_threshold": thr, "test_flag_share": float(pred.mean())}
    out["performance"] = M.evaluate(yte, s, M.f1_threshold(va[target].to_numpy(), sva))
    out["performance_at_review_threshold"] = {k: v for k, v in M.evaluate(yte, s, thr).items() if k in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")}
    out["calibration"] = {"ece_calibrated": ece(yte, s), "ece_uncalibrated": ece(yte, base.predict_proba(Xte)[:, 1]),
                          "brier_calibrated": out["performance"]["brier"]}
    # ---------- fairness (pre-registered tag attributes + region as a descriptive geographic check) ----------
    rep = F.audit(yte, pred, te)
    out["fairness"] = rep
    fairness_plot(rep, None, C.REPORTS / f"fairness_{target}.png")
    if target == C.PRIMARY:
        # region fails the four-fifths screen; mitigate with an equalized-odds post-processor and report the cost
        mp = mitigate_region(bundle, sp, target)
        mf_before = pd.DataFrame(rep["region"]["table"])
        mf_after = F.audit(yte, mp, te)["region"]
        out["mitigation"] = {
            "method": "fairlearn ThresholdOptimizer, equalized_odds across region, fitted on the validation window",
            "before": {"tpr_gap": rep["region"]["tpr_gap"], "fpr_gap": rep["region"]["fpr_gap"], "selection_ratio": rep["region"]["selection_ratio_min_max"],
                       "flag_share": float(pred.mean()), "precision": float(pred[yte == 1].mean() * yte.mean() / max(pred.mean(), 1e-9)), "recall": float(pred[yte == 1].mean())},
            "after": {"tpr_gap": mf_after["tpr_gap"], "fpr_gap": mf_after["fpr_gap"], "selection_ratio": mf_after["selection_ratio_min_max"],
                      "flag_share": float(mp.mean()), "precision": float(mp[yte == 1].mean() * yte.mean() / max(mp.mean(), 1e-9)), "recall": float(mp[yte == 1].mean())},
            "after_table": mf_after["table"], "tags_after": {k: F.audit(yte, mp, te)[k] for k in ("older_american", "servicemember")},
            "balanced_accuracy_before": float(balanced_accuracy_score(yte, pred)), "balanced_accuracy_after": float(balanced_accuracy_score(yte, mp)),
        }
        region_plot(mf_before.to_dict("records"), mf_after["table"], C.REPORTS / "region_mitigation.png")
        eo_pred, eo_cuts, eo_target = equal_opportunity_thresholds(bundle, sp, target)
        eo = F.audit(yte, eo_pred, te)
        out["mitigation_equal_opportunity"] = {
            "method": "per-region thresholds equalising recall (equal opportunity), fitted on the validation window", "recall_target": eo_target,
            "thresholds": eo_cuts, "tpr_gap": eo["region"]["tpr_gap"], "fpr_gap": eo["region"]["fpr_gap"], "selection_ratio": eo["region"]["selection_ratio_min_max"],
            "flag_share": float(eo_pred.mean()), "precision": float(eo_pred[yte == 1].mean() * yte.mean() / max(eo_pred.mean(), 1e-9)),
            "recall": float(eo_pred[yte == 1].mean()), "table": eo["region"]["table"], "tags": {k: eo[k] for k in ("older_american", "servicemember")}}
        region_plot(mf_before.to_dict("records"), eo["region"]["table"], C.REPORTS / "region_equal_opportunity.png")
        # ---------- sensitivity ----------
        out["sensitivity"] = {"perturbations": S.perturbation_table(cal, Xte, yte, thr).round(4).to_dict("records"),
                              "bootstrap": S.bootstrap_ci(yte, s),
                              "quarterly": S.quarterly_stability(te, s, target).round(4).to_dict("records")}
        # retrain the champion without the January 2025 surge weeks
        algo = bundle["algorithm"]
        prm = json.loads((C.REPORTS / f"metrics_{target}.json").read_text())["models"][algo]["cv"]["params"]
        keep = ~S.surge_mask(tr)
        alt = P.MAKERS[algo](**prm).fit(tr.loc[keep, C.RAW_INPUT], tr.loc[keep, target])
        s_alt, v_alt = alt.predict_proba(Xte)[:, 1], alt.predict_proba(va[C.RAW_INPUT])[:, 1]
        thr_alt = M.top_share_threshold(v_alt, C.REVIEW_CAPACITY)
        s_base = base.predict_proba(Xte)[:, 1]
        thr_base = M.top_share_threshold(base.predict_proba(va[C.RAW_INPUT])[:, 1], C.REVIEW_CAPACITY)
        out["sensitivity"]["without_surge"] = {"rows_removed": int((~keep).sum()), "with_surge": S.evaluate_scores(yte, s_base, thr_base),
                                              "without_surge": S.evaluate_scores(yte, s_alt, thr_alt)}
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        pt = pd.DataFrame(out["sensitivity"]["perturbations"])
        ax.barh(pt["scenario"][1:], pt["flag_flip_rate"][1:] * 100, color=TEAL)
        ax.set_xlabel("Share of review flags that change (%)")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(C.REPORTS / "sensitivity_flips.png", dpi=150)
        plt.close(fig)
        fr, mp_ = calibration_curve(yte, s, n_bins=10, strategy="quantile")
        fr0, mp0 = calibration_curve(yte, base.predict_proba(Xte)[:, 1], n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        ax.plot([0, 0.8], [0, 0.8], "--", color="grey")
        ax.plot(mp0, fr0, "o-", color=AMB, label="before calibration")
        ax.plot(mp_, fr, "o-", color=TEAL, label="isotonic (deployed)")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed relief rate")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(C.REPORTS / "calibration.png", dpi=150)
        plt.close(fig)
    (C.REPORTS / f"analysis_{target}.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    import sys
    only = [a for a in sys.argv[1:] if a in C.TARGETS]
    mlflow.set_tracking_uri(C.MLFLOW_URI)
    mlflow.set_experiment(C.EXPERIMENT)
    for t in (only or C.TARGETS):
        o = run_target(t)
        with mlflow.start_run(run_name=f"{t}-fairness-sensitivity-analysis-v2" if only else f"{t}-fairness-sensitivity-analysis"):
            mlflow.set_tags({"target": t, "stage": "analysis"})
            for name in ("older_american", "servicemember"):
                for k in ("disparate_impact_ratio", "equalized_odds_difference", "demographic_parity_difference"):
                    mlflow.log_metric(f"fairness_{name}_{k}", o["fairness"][name][k])
            if "mitigation" in o:
                mlflow.log_metrics({"region_tpr_gap_before": o["mitigation"]["before"]["tpr_gap"], "region_tpr_gap_after": o["mitigation"]["after"]["tpr_gap"],
                                    "region_fpr_gap_before": o["mitigation"]["before"]["fpr_gap"], "region_fpr_gap_after": o["mitigation"]["after"]["fpr_gap"]})
            for f in C.REPORTS.glob(f"*{t}*.png"):
                mlflow.log_artifact(str(f))
            if t == C.PRIMARY:
                for f in ("fairness_monetary_relief.png", "region_mitigation.png", "region_equal_opportunity.png", "sensitivity_flips.png", "calibration.png"):
                    mlflow.log_artifact(str(C.REPORTS / f))
            mlflow.log_artifact(str(C.REPORTS / f"analysis_{t}.json"))
        print(t, "done", flush=True)
