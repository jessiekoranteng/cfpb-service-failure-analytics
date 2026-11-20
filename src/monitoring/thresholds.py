"""Alert rules. Every limit traces to a document: MODEL_CARD.md section 10 (Module 4), ETHICS_CHARTER.md (F2) or the
Module 2 KPI table. `warn` raises a ticket; `hard` pages the model owner and starts the incident process
(docs/final/incident_response_plan.md). Direction 'min' means the value must stay at or above the limit.
"""

import numpy as np

RULES = {
    # signal: (direction, warn, hard, source)
    "roc_auc": ("min", 0.74, 0.70, "Module 4 pre-registered ROC-AUC >= 0.70; warn 0.04 above the limit"),
    "pr_auc_lift": ("min", 2.3, 2.0, "Module 4 pre-registered PR-AUC lift >= 2.0"),
    "recall_at_flag": ("min", 0.48, 0.45, "Module 4 recall at review capacity >= 0.45"),
    "ece": ("max", 0.03, 0.05, "Module 4 alert > 0.03 (warn); Module 2 KPI <= 0.05 (hard), reconciliation C13"),
    "flag_share": ("band", (0.15, 0.25), (0.10, 0.30), "Model Card s10: flagged share outside 15%-25%; hard band is an addition"),
    "unseen_company_share": ("max", 0.05, 0.10, "Model Card s10: unseen values > 5%; hard 10% added (unseen companies fall to ROC-AUC 0.626)"),
    "psi_company": ("max", 0.10, 0.20, "Model Card s10: company-mix PSI > 0.20; 0.10 is the conventional warning level"),
    "psi_score": ("max", 0.10, 0.20, "Score distribution shift, same PSI convention"),
    "region_selection_ratio": ("min", 0.80, 0.70, "Four-fifths rule 0.80 (pre-registered); hard 0.70 added"),
    "f2_tpr_gap": ("max", 0.05, 0.10, "Charter F2 limit 0.05; Module 4 quarterly regional limit 0.10"),
    "f2_fpr_gap": ("max", 0.05, 0.10, "Charter F2 limit 0.05"),
}

MIN_CELL = 10          # a subgroup with fewer positives (or negatives) than this is reported as suppressed, not as a rate
MIN_MONTH_ROWS = 500   # a month with fewer closed complaints than this is not scored for performance


def level(signal: str, value: float | None) -> str:
    """'ok', 'warn' or 'hard' for one observed value. None (not computable) is 'n/a', never 'ok'."""
    if value is None or np.isnan(value):
        return "n/a"
    kind, warn, hard, _ = RULES[signal]
    if kind == "min":
        return "hard" if value < hard else "warn" if value < warn else "ok"
    if kind == "max":
        return "hard" if value > hard else "warn" if value > warn else "ok"
    lo, hi = warn
    hlo, hhi = hard
    return "hard" if (value < hlo or value > hhi) else "warn" if (value < lo or value > hi) else "ok"
