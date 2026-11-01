"""Train, tune and track the risk models with MLflow.

    python -m src.modeling.train                 # full run, tracked in mlflow.db
    python -m src.modeling.train --quick --uri sqlite:////tmp/smoke.db   # smoke test, never the real log

Protocol: time-aware split (see config), TimeSeriesSplit(3) random search on the training window, selection on the
validation window by PR-AUC, test window scored once at the end for all finalists.
"""
import argparse
import json
import time
from datetime import UTC, datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.modeling import config as C
from src.modeling import data as D
from src.modeling import metrics as M
from src.modeling import pipeline as P

SPACES = {
    "logreg": ({"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0], "clf__class_weight": [None, "balanced"]}, 6),
    "forest": ({"clf__n_estimators": [60, 100], "clf__max_depth": [8, 12, 16], "clf__min_samples_leaf": [20, 50, 100],
                "clf__max_features": ["sqrt", 0.5], "clf__class_weight": [None, "balanced_subsample"]}, 5),
    "lightgbm": ({"clf__num_leaves": [15, 31, 63], "clf__learning_rate": [0.03, 0.05, 0.1], "clf__n_estimators": [150, 300, 500],
                  "clf__min_child_samples": [50, 100, 200], "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__reg_lambda": [0.0, 1.0, 5.0],
                  "clf__scale_pos_weight": [1.0, 2.0]}, 10),
}
ALGOS = {C.PRIMARY: ["logreg", "forest", "lightgbm"], C.SECONDARY: ["logreg", "lightgbm"]}
FOREST_ROWS = 100_000


def _fig_curves(y, s, thr, title, path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    RocCurveDisplay.from_predictions(y, s, ax=ax[0])
    ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set_title("ROC")
    PrecisionRecallDisplay.from_predictions(y, s, ax=ax[1])
    ax[1].axhline(float(np.mean(y)), ls="--", color="grey")
    ax[1].set_title("Precision-recall")
    ConfusionMatrixDisplay.from_predictions(y, s >= thr, ax=ax[2], cmap="Blues", colorbar=False)
    ax[2].set_title("Confusion matrix (valid-F1 threshold)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _make_xy(df, target):
    return df[C.RAW_INPUT], df[target].to_numpy()


def tune(algo, Xtr, ytr, quick):
    space, n_iter = SPACES[algo]
    n_iter = 2 if quick else n_iter
    if algo == "forest":
        step = max(1, len(Xtr) // (5_000 if quick else FOREST_ROWS))
        Xs, ys = Xtr.iloc[::step], ytr[::step]
    else:
        Xs, ys = Xtr, ytr
    base = P.MAKERS[algo]()
    search = RandomizedSearchCV(base, space, n_iter=n_iter, scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"}, refit=False,
                                cv=TimeSeriesSplit(n_splits=3), random_state=C.SEED, n_jobs=1, error_score="raise")
    t0 = time.time()
    search.fit(Xs, ys)
    return search, time.time() - t0


def run(targets, quick=False):
    mlflow.set_tracking_uri(C.MLFLOW_URI if not ARGS.uri else ARGS.uri)
    mlflow.set_experiment(C.EXPERIMENT)
    client = MlflowClient()
    a, g = D.load()
    a = D.prepare(a)
    D.assert_no_leakage(C.RAW_INPUT)
    if quick:
        a = a.sample(60_000, random_state=C.SEED)
    results = {}
    with mlflow.start_run(run_name="00-preregistration"):
        mlflow.set_tags({"stage": "preregistration", "preregistered_at": datetime.now(UTC).isoformat(timespec="seconds")})
        mlflow.log_dict(C.CRITERIA, "success_criteria.json")
        mlflow.log_artifact(str(C.ROOT / "docs" / "module4" / "success_criteria.md"))
        mlflow.log_params({"train_end": C.TRAIN_END, "valid_end": C.VALID_END, "window_end": C.WINDOW_END, "capacity": C.REVIEW_CAPACITY})
    for target in targets:
        sp = D.split(a, target)
        sp = {k: D.with_groups(v, g) for k, v in sp.items()}
        Xtr, ytr = _make_xy(sp["train"], target)
        Xva, yva = _make_xy(sp["valid"], target)
        info = {k: {"n": len(v), "prevalence": float(v[target].mean())} for k, v in sp.items()}
        print(target, info, flush=True)
        fitted, rows = {}, []
        for algo in ["majority"] + ALGOS[target]:
            with mlflow.start_run(run_name=f"{target}-{algo}") as parent:
                mlflow.set_tags({"target": target, "algorithm": algo, "stage": "candidate"})
                mlflow.log_params({"target": target, "split": "time-aware", "n_train": len(Xtr), "n_valid": len(Xva), "prevalence_train": float(ytr.mean())})
                if algo == "majority":
                    model = P.make_baseline().fit(Xtr, ytr)
                    cvinfo = {}
                else:
                    search, secs = tune(algo, Xtr, ytr, quick)
                    res = search.cv_results_
                    best = int(np.argmin(res["rank_test_pr_auc"]))
                    for i in range(len(res["params"])):
                        with mlflow.start_run(run_name=f"{target}-{algo}-trial{i}", nested=True):
                            mlflow.set_tag("stage", "tuning-trial")
                            mlflow.log_params({k.replace("clf__", ""): v for k, v in res["params"][i].items()})
                            mlflow.log_metrics({"cv_pr_auc": float(res["mean_test_pr_auc"][i]), "cv_pr_auc_std": float(res["std_test_pr_auc"][i]),
                                                "cv_roc_auc": float(res["mean_test_roc_auc"][i]), "cv_roc_auc_std": float(res["std_test_roc_auc"][i])})
                    bp = search.cv_results_["params"][best]
                    mlflow.log_params({k.replace("clf__", "best_"): v for k, v in bp.items()})
                    mlflow.log_metrics({"cv_best_pr_auc": float(res["mean_test_pr_auc"][best]), "tuning_seconds": secs})
                    cvinfo = {"cv_pr_auc": float(res["mean_test_pr_auc"][best]), "cv_pr_auc_std": float(res["std_test_pr_auc"][best]), "params": {k.replace("clf__", ""): v for k, v in bp.items()}}
                    model = P.MAKERS[algo](**{k.replace("clf__", ""): v for k, v in bp.items()})
                    if algo == "forest" and not quick:
                        model.set_params(clf__n_jobs=2)
                    t0 = time.time()
                    model.fit(Xtr, ytr)
                    mlflow.log_metric("fit_seconds", time.time() - t0)
                sva = model.predict_proba(Xva)[:, 1]
                thr = M.f1_threshold(yva, sva) if algo != "majority" else 0.5
                mv = M.evaluate(yva, sva, thr)
                mlflow.log_metrics({f"valid_{k}": v for k, v in mv.items() if isinstance(v, float)})
                _fig_curves(yva, sva, thr, f"{target} / {algo}: validation window", C.REPORTS / f"curves_valid_{target}_{algo}.png")
                mlflow.log_artifact(str(C.REPORTS / f"curves_valid_{target}_{algo}.png"))
                mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
                fitted[algo] = {"model": model, "run_id": parent.info.run_id, "threshold": thr, "valid": mv, "cv": cvinfo}
                rows.append((algo, mv["pr_auc"]))
                print(f"  {algo}: valid ROC {mv['roc_auc']:.4f} PR {mv['pr_auc']:.4f} (cv {cvinfo.get('cv_pr_auc')})", flush=True)
        champion = max((a_ for a_ in fitted if a_ != "majority"), key=lambda k: fitted[k]["valid"]["pr_auc"])
        # ---- test window: touched once, after selection ----
        te_x, te_y = _make_xy(sp["test"], target)
        out = {"target": target, "split": info, "champion": champion, "models": {}}
        for algo, f in fitted.items():
            ste = f["model"].predict_proba(te_x)[:, 1]
            mt = M.evaluate(te_y, ste, f["threshold"])
            _fig_curves(te_y, ste, f["threshold"], f"{target} / {algo}: TEST window", C.REPORTS / f"curves_test_{target}_{algo}.png")
            with mlflow.start_run(run_id=f["run_id"]):
                mlflow.log_metrics({f"test_{k}": v for k, v in mt.items() if isinstance(v, float)})
                mlflow.log_artifact(str(C.REPORTS / f"curves_test_{target}_{algo}.png"))
            out["models"][algo] = {"valid": f["valid"], "test": mt, "cv": f["cv"], "run_id": f["run_id"]}
        # ---- calibrate the champion on the validation window and register it ----
        best = fitted[champion]
        cal = CalibratedClassifierCV(FrozenEstimator(best["model"]), method="isotonic").fit(Xva, yva)
        sc = cal.predict_proba(te_x)[:, 1]
        out["champion_calibrated_test"] = M.evaluate(te_y, sc, M.f1_threshold(yva, cal.predict_proba(Xva)[:, 1]))
        C.MODELS.mkdir(exist_ok=True)
        joblib.dump({"calibrated": cal, "base": best["model"], "target": target, "features": C.RAW_INPUT, "algorithm": champion,
                     "review_threshold": float(np.quantile(cal.predict_proba(Xva)[:, 1], 1 - C.REVIEW_CAPACITY)),
                     "trained_at": datetime.now(UTC).isoformat(timespec="seconds")}, C.MODELS / f"risk_{target}.joblib")
        with mlflow.start_run(run_name=f"{target}-champion-calibrated") as _:
            mlflow.set_tags({"target": target, "algorithm": champion, "stage": "champion"})
            mlflow.log_metrics({f"test_{k}": v for k, v in out["champion_calibrated_test"].items() if isinstance(v, float)})
            mlflow.sklearn.log_model(cal, name="model", serialization_format="cloudpickle", registered_model_name=f"cfpb-{target}-risk")
            mlflow.log_artifact(str(C.MODELS / f"risk_{target}.joblib"))
        ver = max(int(v.version) for v in client.search_model_versions(f"name='cfpb-{target}-risk'"))
        client.set_registered_model_alias(f"cfpb-{target}-risk", "champion", str(ver))
        client.set_model_version_tag(f"cfpb-{target}-risk", str(ver), "algorithm", champion)
        out["registry"] = {"name": f"cfpb-{target}-risk", "version": ver, "alias": "champion"}
        results[target] = out
        (C.REPORTS / f"metrics_{target}.json").write_text(json.dumps(out, indent=2, default=float))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--uri", default=None)
    ap.add_argument("--targets", nargs="+", default=C.TARGETS)
    ARGS = ap.parse_args()
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    run(ARGS.targets, ARGS.quick)
