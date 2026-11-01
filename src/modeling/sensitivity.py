"""Robustness checks: input perturbations, bootstrap confidence intervals, time-slice stability and a retrain without the surge."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.modeling import config as C
from src.modeling import metrics as M


def _flags(scores, thr):
    return scores >= thr


def perturbation_table(model, X: pd.DataFrame, y, thr: float) -> pd.DataFrame:
    """How far do scores, AUC and review flags move when inputs are nudged or degraded?"""
    base = model.predict_proba(X)[:, 1]
    base_flag = _flags(base, thr)
    scenarios = {
        "company volume -20%": lambda d: d.assign(company_volume_lag4=d["company_volume_lag4"] * 0.8),
        "company volume +20%": lambda d: d.assign(company_volume_lag4=d["company_volume_lag4"] * 1.2),
        "company volume missing": lambda d: d.assign(company_volume_lag4=np.nan),
        "sub-issue not recorded": lambda d: d.assign(**{"Sub-issue": "Unknown"}),
        "channel set to Web": lambda d: d.assign(**{"Submitted via": "Web"}),
        "month shifted +1": lambda d: d.assign(month=(d["month"] % 12) + 1),
        "unseen company": lambda d: d.assign(company_key="NEW ENTRANT BANK"),
    }
    rows = [{"scenario": "baseline", "roc_auc": roc_auc_score(y, base), "pr_auc": average_precision_score(y, base), "mean_abs_score_change": 0.0, "flag_flip_rate": 0.0}]
    for name, fn in scenarios.items():
        s = model.predict_proba(fn(X.copy()))[:, 1]
        rows.append({"scenario": name, "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
                     "mean_abs_score_change": float(np.mean(np.abs(s - base))), "flag_flip_rate": float(np.mean(_flags(s, thr) != base_flag))})
    return pd.DataFrame(rows)


def bootstrap_ci(y, scores, n_boot=300, seed=C.SEED) -> dict:
    rng = np.random.default_rng(seed)
    y, scores = np.asarray(y), np.asarray(scores)
    aucs, prs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc_score(y[i], scores[i]))
        prs.append(average_precision_score(y[i], scores[i]))
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return {"roc_auc_ci": q(aucs), "pr_auc_ci": q(prs), "n_boot": len(aucs)}


def quarterly_stability(df: pd.DataFrame, scores, target) -> pd.DataFrame:
    d = pd.DataFrame({"q": df["week_start"].dt.to_period("Q").astype(str).to_numpy(), "y": df[target].to_numpy(), "s": np.asarray(scores)})
    rows = []
    for q, g in d.groupby("q"):
        if g["y"].nunique() < 2:
            continue
        rows.append({"quarter": q, "n": len(g), "prevalence": g["y"].mean(), "roc_auc": roc_auc_score(g["y"], g["s"]), "pr_auc": average_precision_score(g["y"], g["s"])})
    return pd.DataFrame(rows)


def surge_mask(df: pd.DataFrame) -> np.ndarray:
    """Weeks of the January 2025 mass-complaint burst (about 66,000 complaints, 0% monetary relief for the top company)."""
    return ((df["week_start"] >= "2025-01-13") & (df["week_start"] <= "2025-01-27")).to_numpy()


def evaluate_scores(y, s, thr):
    m = M.evaluate(y, s, thr)
    return {k: m[k] for k in ("roc_auc", "pr_auc", "recall_at_capacity", "precision_at_capacity", "brier")}
