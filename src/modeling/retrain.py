"""Retraining: when to retrain, how a challenger is built, the gates it must pass, and who may promote it.

    python -m src.modeling.retrain decide    --monitoring monitoring_app/assets/monitoring_data.json
    python -m src.modeling.retrain challenger [--dry-run]             # builds, gates and registers a challenger (alias `challenger`)
    python -m src.modeling.retrain promote --version N --approver "Name, role" --ticket ID

Rules that cannot be bypassed from the command line:
  * the `champion` alias and the deployed bundle in models/ never change during `challenger`;
  * a challenger that fails any gate cannot be promoted;
  * a version tagged dry_run=true (built on historical windows to prove the pipeline) can never be promoted;
  * promotion needs a named human approver and a ticket id, and archives the bundle it replaces.
The primary (monetary-relief) model only. The late-response model is advisory and unstable (ROC-AUC 0.759 to 0.892 across
windows), so it is not retrained automatically.
"""
import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, roc_auc_score

from src.modeling import config as C
from src.monitoring import metrics as MM

MODEL_NAME = f"cfpb-{C.PRIMARY}-risk"
PERF_SIGNALS = ("roc_auc", "pr_auc_lift", "recall_at_flag", "ece")
SCHEDULED_MONTHS = 6          # retrain at least twice a year even if nothing alarms
# Dry-run windows: fit through 2025-09, calibrate on Q4 2025, gate on Jan to Jun 2026 (all unseen by the deployed champion).
DRY_WINDOWS = {"fit_end": "2025-09-30", "calib_end": "2025-12-31", "gate_end": C.WINDOW_END}
F2_SLACK = 0.01               # a challenger may be this much worse than the champion on a fairness gap if it stays within 0.05


# ---------------------------------------------------------------- 1. when to retrain
def decide(mon: dict, trained_at: str | None = None, today: datetime | None = None) -> dict:
    """Turn the monitoring output into a decision with reasons. Never retrains by itself."""
    today = today or datetime.now(UTC)
    months = mon["monthly"]
    reasons, escalate = [], []
    for sig in PERF_SIGNALS:
        lv = [r["levels"].get(sig) for r in months]
        if "hard" in lv:
            reasons.append(f"{sig}: hard breach in {months[lv.index('hard')]['month']}")
        elif any(a == b == "warn" for a, b in itertools.pairwise(lv)):
            reasons.append(f"{sig}: warning in two consecutive months")
    last3 = months[-3:]
    if sum(r["levels"].get("psi_company") == "hard" for r in last3) >= 2:
        reasons.append("psi_company: hard in at least 2 of the last 3 months")
    dr = mon["drift"]
    if dr["drifted_columns"]["share"] >= 0.5:
        reasons.append(f"Evidently: {int(dr['drifted_columns']['count'])} of {len(dr['columns'])} columns drifted")
    for k, g in mon["fairness_tags"]["replay_window"].items():
        if g["confirmed_breach"]:
            escalate.append(f"F2 confirmed breach for {k} (interval wholly above the 0.05 charter limit)")
    if trained_at:
        age = (today - datetime.fromisoformat(trained_at)).days / 30.4
        if age >= SCHEDULED_MONTHS:
            reasons.append(f"scheduled: champion is {age:.1f} months old (limit {SCHEDULED_MONTHS})")
    watch = [f"{a['signal']} {a['level']} in {a['month']}" for a in mon["alerts"] if not a["noisy"]]
    if escalate:
        decision = "ESCALATE to the AI Ethics and Data Governance Review Panel"
    elif reasons:
        decision = "RETRAIN RECOMMENDED"
    else:
        decision = "NO RETRAIN TRIGGERED"
    summary = ("; ".join(escalate + reasons) if (escalate or reasons) else "No trigger fired.") + (f" Watch items: {', '.join(watch)}." if watch else "")
    return {"decision": decision, "reasons": reasons, "escalations": escalate, "watch": watch, "summary": summary,
            "evaluated_at": today.isoformat(timespec="seconds"), "champion_trained_at": trained_at}


# ---------------------------------------------------------------- 2. building a challenger
def build_challenger(df: pd.DataFrame, champion: dict, windows: dict, target: str = C.PRIMARY) -> dict:
    """Same algorithm and hyper-parameters as the champion (cloned), refitted on newer data, calibrated on a later window.
    `df` is the prepared analysis table with `week_start` and the target. Returns a bundle shaped like the deployed one."""
    from src.modeling.analysis import equal_opportunity_thresholds
    d = df[df["week_start"] <= pd.Timestamp(windows["gate_end"])]
    d = d[d[target].notna()]
    fit = d[d["week_start"] <= pd.Timestamp(windows["fit_end"])]
    cal = d[(d["week_start"] > pd.Timestamp(windows["fit_end"])) & (d["week_start"] <= pd.Timestamp(windows["calib_end"]))]
    base = clone(champion["base"]).fit(fit[C.RAW_INPUT], fit[target].astype(int))
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(cal[C.RAW_INPUT], cal[target].astype(int))
    sc = calibrated.predict_proba(cal[C.RAW_INPUT])[:, 1]
    bundle = {"calibrated": calibrated, "base": base, "target": target, "features": C.RAW_INPUT, "algorithm": champion["algorithm"],
              "review_threshold": float(np.quantile(sc, 1 - C.REVIEW_CAPACITY)), "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "windows": windows, "n_fit": len(fit), "n_calib": len(cal)}
    _, cuts, _ = equal_opportunity_thresholds(bundle, {"valid": cal, "test": cal.head(1)}, target)
    bundle["region_thresholds"] = cuts
    return bundle


# ---------------------------------------------------------------- 3. gates
def evaluate_bundle(bundle: dict, d: pd.DataFrame, groups: pd.DataFrame | None = None) -> dict:
    """Everything the gates need, on one window, at the DEPLOYED rule (region thresholds)."""
    from src.modeling.charter import deployed_flags
    s = bundle["calibrated"].predict_proba(d[C.RAW_INPUT])[:, 1]
    y = d[bundle["target"]].astype(int).to_numpy()
    flag = deployed_flags(bundle, d, s)
    out = {"n": len(y), "relief_rate": float(y.mean()), "roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
           "ece": MM.ece(y, s), "flag_share": float(flag.mean()), "recall_at_flag": float(flag[y == 1].mean()),
           "region_selection_ratio": MM.selection_ratio(flag, d["region"].astype(str))}
    out["pr_auc_lift"] = out["pr_auc"] / out["relief_rate"]
    if groups is not None:
        for name, col in (("older_american", "tag_older"), ("servicemember", "tag_servicemember")):
            g = MM.vs_rest_gaps(y, flag, groups[col].astype(bool).to_numpy())
            out[f"f2_{name}"] = {"tpr_gap": g["tpr_gap"], "fpr_gap": g["fpr_gap"]}
    return out


def shap_importance(bundle: dict, X: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP value per model feature on a sample (same transform path as the API's reason codes)."""
    import shap
    Xt = bundle["base"][:-1].transform(X)
    sv = shap.TreeExplainer(bundle["base"][-1]).shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else np.asarray(sv)
    sv = sv if sv.ndim == 2 else sv[:, :, 1]
    return pd.Series(np.abs(sv).mean(axis=0), index=list(Xt.columns))


def explanation_stability(champion: dict, challenger: dict, X: pd.DataFrame, n: int = 2000) -> float:
    """Spearman correlation of the two models' global SHAP importances (Module 2 acceptance check: at least 0.8)."""
    from scipy.stats import spearmanr
    s = X.sample(min(n, len(X)), random_state=C.SEED)
    a, b = shap_importance(champion, s), shap_importance(challenger, s)
    return float(spearmanr(a.loc[a.index], b.loc[a.index]).statistic)


def run_gates(champ: dict, chall: dict, features: list[str], stability: float) -> dict:
    """Each gate returns pass/fail and the numbers behind it. All gates must pass; nothing is averaged."""
    from src.modeling.data import assert_no_leakage
    gates = []

    def add(name, ok, detail):
        gates.append({"gate": name, "passed": bool(ok), "detail": detail})

    add("performance: ROC-AUC >= 0.70 and within 0.01 of champion", chall["roc_auc"] >= 0.70 and chall["roc_auc"] >= champ["roc_auc"] - 0.01,
        f"challenger {chall['roc_auc']:.4f} vs champion {champ['roc_auc']:.4f}")
    add("performance: PR-AUC within 0.01 of champion", chall["pr_auc"] >= champ["pr_auc"] - 0.01, f"challenger {chall['pr_auc']:.4f} vs champion {champ['pr_auc']:.4f}")
    add("calibration: ECE <= 0.05", chall["ece"] <= 0.05, f"challenger {chall['ece']:.4f} vs champion {champ['ece']:.4f}")
    add("capacity: flagged share within 15% to 25%", 0.15 <= chall["flag_share"] <= 0.25, f"{chall['flag_share']:.3f}")
    add("fairness: regional selection ratio >= 0.80", (chall["region_selection_ratio"] or 0) >= 0.80, f"{chall['region_selection_ratio']:.3f} (champion {champ['region_selection_ratio']:.3f})")
    for g in ("older_american", "servicemember"):
        for k in ("tpr_gap", "fpr_gap"):
            c, h = champ[f"f2_{g}"][k], chall[f"f2_{g}"][k]
            ok = h is not None and c is not None and (h <= 0.05 or h <= c + F2_SLACK)
            add(f"fairness F2: {g} {k} within 0.05 or no worse than champion + {F2_SLACK}", ok, f"challenger {h:.4f} vs champion {c:.4f}")
    add("explanation stability: global SHAP ranking Spearman >= 0.80 vs champion", stability >= 0.80, f"{stability:.3f}")
    try:
        assert_no_leakage(features)
        add("no forbidden or protected input", True, "feature list checked")
    except ValueError as e:
        add("no forbidden or protected input", False, str(e))
    add("input contract unchanged", list(features) == list(C.RAW_INPUT), f"{len(features)} fields")
    return {"passed": all(g["passed"] for g in gates), "gates": gates}


# ---------------------------------------------------------------- 4. registry
def _mlflow(uri):
    import mlflow
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(C.EXPERIMENT)
    return mlflow


def register_challenger(bundle: dict, report: dict, uri: str, dry_run: bool, models_dir: Path) -> int:
    mlflow = _mlflow(uri)
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    path = Path(models_dir) / "challenger" / f"risk_{C.PRIMARY}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    with mlflow.start_run(run_name=f"{C.PRIMARY}-challenger"):
        mlflow.set_tags({"target": C.PRIMARY, "stage": "challenger", "dry_run": str(dry_run).lower(), "gates_passed": str(report["passed"]).lower()})
        mlflow.log_params({**{f"window_{k}": v for k, v in bundle["windows"].items()}, "n_fit": bundle["n_fit"], "n_calib": bundle["n_calib"]})
        mlflow.log_metrics({f"challenger_{k}": v for k, v in report["challenger"].items() if isinstance(v, float)})
        mlflow.log_metrics({f"champion_{k}": v for k, v in report["champion"].items() if isinstance(v, float)})
        mlflow.log_dict(report, "retrain_gate_report.json")
        mlflow.sklearn.log_model(bundle["calibrated"], name="model", serialization_format="cloudpickle", registered_model_name=MODEL_NAME)
    ver = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
    client.set_registered_model_alias(MODEL_NAME, "challenger", str(ver))
    for k, v in {"dry_run": str(dry_run).lower(), "gates_passed": str(report["passed"]).lower(), "bundle": str(path)}.items():
        client.set_model_version_tag(MODEL_NAME, str(ver), k, v)
    return ver


def promote(version: int, approver: str, ticket: str, uri: str, models_dir: Path) -> dict:
    """Make a gated challenger the champion. Refuses without a named approver and ticket, a failed gate or a dry-run version."""
    if not approver.strip() or not ticket.strip():
        raise PermissionError("promotion needs a named approver and a ticket id")
    _mlflow(uri)
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    mv = client.get_model_version(MODEL_NAME, str(version))
    tags = mv.tags
    if tags.get("dry_run") == "true":
        raise PermissionError(f"version {version} was built as a dry run on historical windows and can never be promoted")
    if tags.get("gates_passed") != "true":
        raise PermissionError(f"version {version} did not pass every gate")
    src = Path(tags["bundle"])
    live = Path(models_dir) / f"risk_{C.PRIMARY}.joblib"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    if live.exists():
        arch = Path(models_dir) / "archive"
        arch.mkdir(exist_ok=True)
        shutil.copy2(live, arch / f"risk_{C.PRIMARY}_{stamp}.joblib")
    shutil.copy2(src, live)
    old = str(client.get_model_version_by_alias(MODEL_NAME, "champion").version) if _has_alias(client, "champion") else None
    client.set_registered_model_alias(MODEL_NAME, "champion", str(version))
    for k, v in {"promoted_by": approver, "promotion_ticket": ticket, "promoted_at": stamp, "replaced_version": str(old)}.items():
        client.set_model_version_tag(MODEL_NAME, str(version), k, v)
    return {"promoted": version, "replaced": old, "approver": approver, "ticket": ticket, "archived_to": f"archive/risk_{C.PRIMARY}_{stamp}.joblib"}


def _has_alias(client, alias):
    try:
        client.get_model_version_by_alias(MODEL_NAME, alias)
        return True
    except Exception:  # noqa: BLE001  (MLflow raises a generic exception for a missing alias)
        return False


# ---------------------------------------------------------------- CLI
def cmd_challenger(args):
    from src.modeling import data as D
    a, g = D.load()
    a = D.prepare(a)
    a = D.with_groups(a, g)
    champion = joblib.load(C.MODELS / f"risk_{C.PRIMARY}.joblib")
    windows = DRY_WINDOWS if args.dry_run else {"fit_end": args.fit_end, "calib_end": args.calib_end, "gate_end": args.gate_end}
    if not args.dry_run and not all(windows.values()):
        raise SystemExit("a real run needs --fit-end, --calib-end and --gate-end; use --dry-run to rehearse on historical windows")
    chall = build_challenger(a, champion, windows)
    gate = a[(a["week_start"] > pd.Timestamp(windows["calib_end"])) & (a["week_start"] <= pd.Timestamp(windows["gate_end"])) & a[C.PRIMARY].notna()]
    report = {"windows": windows, "dry_run": bool(args.dry_run), "champion": evaluate_bundle(champion, gate, gate), "challenger": evaluate_bundle(chall, gate, gate)}
    report["explanation_stability_spearman"] = explanation_stability(champion, chall, gate[C.RAW_INPUT])
    report.update(run_gates(report["champion"], report["challenger"], chall["features"], report["explanation_stability_spearman"]))
    ver = register_challenger(chall, report, args.uri, args.dry_run, C.MODELS)
    report["registered_version"], report["alias"] = ver, "challenger"
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"challenger v{ver} registered (dry_run={args.dry_run}); gates passed: {report['passed']}")
    for gt in report["gates"]:
        print(("PASS " if gt["passed"] else "FAIL ") + gt["gate"] + " | " + gt["detail"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--monitoring", default="monitoring_app/assets/monitoring_data.json")
    d.add_argument("--out", default="monitoring_app/assets/retrain_decision.json")
    c = sub.add_parser("challenger")
    c.add_argument("--dry-run", action="store_true")
    for flag in ("--fit-end", "--calib-end", "--gate-end"):
        c.add_argument(flag)
    c.add_argument("--uri", default=C.MLFLOW_URI)
    c.add_argument("--report", default=str(C.REPORTS.parent / "final" / "retrain_dry_run.json"))
    p = sub.add_parser("promote")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--approver", required=True)
    p.add_argument("--ticket", required=True)
    p.add_argument("--uri", default=C.MLFLOW_URI)
    a = ap.parse_args()
    if a.cmd == "decide":
        mon = json.loads(Path(a.monitoring).read_text())
        res = decide(mon, joblib.load(C.MODELS / f"risk_{C.PRIMARY}.joblib")["trained_at"])
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(res["decision"], "|", res["summary"])
    elif a.cmd == "challenger":
        cmd_challenger(a)
    else:
        print(json.dumps(promote(a.version, a.approver, a.ticket, a.uri, C.MODELS), indent=1))


if __name__ == "__main__":
    main()
