"""Paired bootstrap: is the champion really different from the logistic-regression baseline on the test window?

The champion was chosen on validation PR-AUC and the test window is scored once, so this does NOT re-select a model.
It only measures how large the champion-minus-baseline gap is relative to sampling noise, using the same resampled
complaints for both models (paired), so the report can state whether the pre-registered "beat the logistic baseline"
criterion is missed by a real margin or by noise.

Run:  python -m src.modeling.compare
"""
import json

import joblib
import mlflow
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.modeling import config as C
from src.modeling import data

N_BOOT = 300


def recall_at_capacity(y, s, share=C.REVIEW_CAPACITY):
    k = max(1, round(share * len(y)))
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].sum() / max(1, y.sum()))


def paired_bootstrap(y, s_a, s_b, n_boot=N_BOOT, seed=C.SEED):
    """Return point estimates and 95% percentile CIs of (A - B) for PR-AUC, ROC-AUC and recall@capacity."""
    rng = np.random.default_rng(seed)
    n = len(y)
    fns = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score, "recall_at_capacity": recall_at_capacity}
    point = {k: float(f(y, s_a) - f(y, s_b)) for k, f in fns.items()}
    draws = {k: [] for k in fns}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        for k, f in fns.items():
            draws[k].append(f(y[i], s_a[i]) - f(y[i], s_b[i]))
    out = {}
    for k, d in draws.items():
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[k] = {"difference": point[k], "ci95": [float(lo), float(hi)], "ci_includes_zero": bool(lo <= 0 <= hi)}
    return out


def main(target=C.PRIMARY):
    mlflow.set_tracking_uri(C.MLFLOW_URI)
    a, _ = data.load()
    parts = data.split(data.prepare(a), target)
    te = parts["test"]
    X, y = te[C.RAW_INPUT], te[target].to_numpy()
    champ = joblib.load(C.MODELS / f"risk_{target}.joblib")
    metrics = json.loads((C.REPORTS / f"metrics_{target}.json").read_text())
    lr = mlflow.sklearn.load_model(f"runs:/{metrics['models']['logreg']['run_id']}/model")
    s_champ = champ["calibrated"].predict_proba(X)[:, 1]
    s_lr = lr.predict_proba(X)[:, 1]
    res = {"target": target, "n_test": len(y), "n_boot": N_BOOT, "champion": champ["algorithm"] + " (calibrated)",
           "baseline": "logistic regression", **paired_bootstrap(y, s_champ, s_lr)}
    (C.REPORTS / f"champion_vs_logreg_{target}.json").write_text(json.dumps(res, indent=2))
    for k in ("pr_auc", "roc_auc", "recall_at_capacity"):
        r = res[k]
        print(f"{k:19s} champion-minus-logreg {r['difference']:+.4f}  95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
    return res


def capacity_check():
    """Recall/precision when EXACTLY the top 20% of test complaints by score are flagged, plus the share a validation-fixed
    threshold would flag on test (they differ when the score distribution drifts, as it does for the late-response model)."""
    mlflow.set_tracking_uri(C.MLFLOW_URI)
    a, _ = data.load()
    d = data.prepare(a)
    out = {}
    for target in C.TARGETS:
        te = data.split(d, target)["test"]
        y = te[target].to_numpy()
        b = joblib.load(C.MODELS / f"risk_{target}.joblib")
        s = b["calibrated"].predict_proba(te[C.RAW_INPUT])[:, 1]
        k = round(C.REVIEW_CAPACITY * len(y))
        top = np.argsort(-s, kind="stable")[:k]
        cap_thr = json.loads((C.REPORTS / f"analysis_{target}.json").read_text())["performance"]["capacity_threshold"]
        out[target] = {"n_test": len(y), "flagged": k, "recall_top20": float(y[top].sum() / y.sum()), "precision_top20": float(y[top].sum() / k),
                       "validation_fixed_threshold": float(cap_thr), "share_flagged_by_fixed_threshold": float((s >= cap_thr).mean())}
        print(target, out[target])
    (C.REPORTS / "capacity_check.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
    capacity_check()
