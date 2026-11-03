"""Fairness metrics for the Ethical Compliance dashboard, computed with Fairlearn from published aggregates.

The deployed dashboard holds no complaint-level rows. A group's confusion counts (true/false positives and negatives) carry
everything the rate metrics need, so each count is expanded into one weighted row and handed to Fairlearn's MetricFrame
with `sample_weight`. The numbers are therefore exactly what Fairlearn would return on the full table.
"""
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)

CELLS = (("tn", 0, 0), ("fp", 0, 1), ("fn", 1, 0), ("tp", 1, 1))


def counts_from_hist(hist: dict, threshold: float, bin_width: float) -> dict:
    """Confusion counts per group when everything scoring >= threshold is flagged. Exact when the threshold is a multiple of bin_width."""
    k = round(threshold / bin_width)
    out = {}
    for g, h in hist.items():
        neg, pos = np.asarray(h["neg"]), np.asarray(h["pos"])
        out[g] = {"tn": int(neg[:k].sum()), "fp": int(neg[k:].sum()), "fn": int(pos[:k].sum()), "tp": int(pos[k:].sum())}
    return out


def expand(counts: dict) -> pd.DataFrame:
    rows = [{"group": g, "y_true": yt, "y_pred": yp, "w": c[cell]} for g, c in counts.items() for cell, yt, yp in CELLS if c[cell] > 0]
    return pd.DataFrame(rows)


def metric_frame(counts: dict) -> MetricFrame:
    d = expand(counts)
    w = d["w"].to_numpy(dtype=float)
    return MetricFrame(metrics={"flagged": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate},
                       y_true=d["y_true"], y_pred=d["y_pred"], sensitive_features=d["group"],
                       sample_params={"flagged": {"sample_weight": w}, "tpr": {"sample_weight": w}, "fpr": {"sample_weight": w}})


def summarise(counts: dict) -> dict:
    """Per-group table plus the group-level fairness measures used in the pre-registered limits and the Ethical AI Charter."""
    d = expand(counts)
    w = d["w"].to_numpy(dtype=float)
    mf = metric_frame(counts)
    t = mf.by_group.copy()
    t["complaints"] = {g: sum(c.values()) for g, c in counts.items()}
    t["actual_relief_rate"] = {g: (c["fn"] + c["tp"]) / sum(c.values()) for g, c in counts.items()}
    kw = {"y_true": d["y_true"], "y_pred": d["y_pred"], "sensitive_features": d["group"], "sample_weight": w}
    return {
        "table": t.reset_index().rename(columns={"group": "Group"}),
        "disparate_impact_ratio": float(demographic_parity_ratio(**kw)),
        "demographic_parity_difference": float(demographic_parity_difference(**kw)),
        "equalized_odds_difference": float(equalized_odds_difference(**kw)),
        "tpr_gap": float(t["tpr"].max() - t["tpr"].min()),
        "fpr_gap": float(t["fpr"].max() - t["fpr"].min()),
        "flagged_share": float(sum(c["fp"] + c["tp"] for c in counts.values()) / sum(sum(c.values()) for c in counts.values())),
    }


def verdicts(s: dict, limits: dict) -> list:
    """Pass/fail against the pre-registered limits (disparate impact >= 0.80, equalized-odds and parity differences <= 0.10)
    and the Charter F2 limit (true- and false-positive-rate gaps <= 0.05)."""
    return [
        ("Disparate impact ratio", s["disparate_impact_ratio"], f">= {limits['disparate_impact_min']:.2f}", s["disparate_impact_ratio"] >= limits["disparate_impact_min"]),
        ("Equalized-odds difference", s["equalized_odds_difference"], f"<= {limits['equalized_odds_diff_max']:.2f}", s["equalized_odds_difference"] <= limits["equalized_odds_diff_max"]),
        ("Demographic-parity difference", s["demographic_parity_difference"], f"<= {limits['demographic_parity_diff_max']:.2f}", s["demographic_parity_difference"] <= limits["demographic_parity_diff_max"]),
        ("Charter F2: true-positive-rate gap", s["tpr_gap"], f"<= {limits['charter_f2']:.2f}", s["tpr_gap"] <= limits["charter_f2"]),
        ("Charter F2: false-positive-rate gap", s["fpr_gap"], f"<= {limits['charter_f2']:.2f}", s["fpr_gap"] <= limits["charter_f2"]),
    ]
