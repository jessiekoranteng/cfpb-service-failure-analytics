"""Fairness audit and mitigation (Fairlearn). Protected flags come from the restricted audit table and are used for
evaluation (and for the mitigation constraint during training) only. They are never model inputs."""
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)

from src.modeling import config as C


def _binary_report(y, pred, flag) -> dict:
    """One protected attribute with two groups (flagged group vs everyone else)."""
    y, pred = np.asarray(y), np.asarray(pred).astype(int)
    s = np.asarray(flag).astype(int)
    mf = MetricFrame(metrics={"selection": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate, "n": count},
                     y_true=y, y_pred=pred, sensitive_features=s)
    by = mf.by_group
    sel = by["selection"]
    di = float(sel.min() / sel.max()) if sel.max() > 0 else float("nan")
    out = {"n_group": int(by.loc[1, "n"]), "n_rest": int(by.loc[0, "n"]),
           "base_rate_group": float(y[s == 1].mean()), "base_rate_rest": float(y[s == 0].mean()),
           "selection_group": float(sel.loc[1]), "selection_rest": float(sel.loc[0]),
           "tpr_group": float(by.loc[1, "tpr"]), "tpr_rest": float(by.loc[0, "tpr"]),
           "fpr_group": float(by.loc[1, "fpr"]), "fpr_rest": float(by.loc[0, "fpr"]),
           "demographic_parity_difference": float(demographic_parity_difference(y, pred, sensitive_features=s)),
           "demographic_parity_ratio": float(demographic_parity_ratio(y, pred, sensitive_features=s)),
           "equalized_odds_difference": float(equalized_odds_difference(y, pred, sensitive_features=s)),
           "disparate_impact_ratio": di}
    crit = C.CRITERIA["fairness"]
    out["passes"] = {"disparate_impact": bool(di >= crit["disparate_impact_min"]),
                     "equalized_odds": bool(out["equalized_odds_difference"] <= crit["equalized_odds_diff_max"]),
                     "demographic_parity": bool(out["demographic_parity_difference"] <= crit["demographic_parity_diff_max"])}
    return out


def audit(y, pred, groups: pd.DataFrame) -> dict:
    """Binary audits for each protected attribute, plus the 4-way tag group and region as descriptive views."""
    rep = {name: _binary_report(y, pred, groups[col]) for name, col in C.PROTECTED.items()}
    pred = np.asarray(pred).astype(int)
    for col in ("tag_group", "region"):
        mf = MetricFrame(metrics={"selection": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate, "n": count},
                         y_true=np.asarray(y), y_pred=pred, sensitive_features=groups[col].astype(str).to_numpy())
        t = mf.by_group.round(4)
        rep[col] = {"table": t.reset_index().rename(columns={"sensitive_feature_0": col}).to_dict(orient="records"),
                    "selection_ratio_min_max": float(t["selection"].min() / t["selection"].max()),
                    "tpr_gap": float(t["tpr"].max() - t["tpr"].min()), "fpr_gap": float(t["fpr"].max() - t["fpr"].min())}
    return rep


def flag_top(scores, share=C.REVIEW_CAPACITY, threshold=None):
    scores = np.asarray(scores)
    t = threshold if threshold is not None else float(np.quantile(scores, 1 - share))
    return (scores >= t).astype(int)


class WeightedPipeline:
    """Adapter: Fairlearn's reductions call fit(X, y, sample_weight); route the weight to the last pipeline step."""

    def __init__(self, make_pipeline):
        self.make_pipeline = make_pipeline

    def fit(self, X, y, sample_weight=None):
        self.pipe_ = self.make_pipeline()
        self.pipe_.fit(X, y, **({"clf__sample_weight": sample_weight} if sample_weight is not None else {}))
        self.classes_ = self.pipe_.classes_
        return self

    def predict(self, X):
        return self.pipe_.predict(X)

    def predict_proba(self, X):
        return self.pipe_.predict_proba(X)
