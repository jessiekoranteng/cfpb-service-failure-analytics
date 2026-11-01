"""Module 4 modelling configuration: target, split, features and pre-registered success criteria.

Everything a reviewer needs to reproduce a run lives here. Values were fixed BEFORE the first model was trained
(see docs/module4/success_criteria.md and the `preregistered_at` tag on the first MLflow run).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "m4"
MODELS = ROOT / "models"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT = "cfpb-service-failure-risk"
SEED = 42

# Targets (Module 3 labels). Prediction is made at intake: everything below is known when a complaint is received.
PRIMARY = "monetary_relief"      # G1: will the company close the case with monetary relief?
SECONDARY = "untimely"           # G2: will the company miss the 15-day response window?
TARGETS = [PRIMARY, SECONDARY]

CATEGORICAL = ["Product", "Sub-product", "Issue", "Sub-issue", "Submitted via", "region", "company_key"]
NUMERIC = ["company_volume_lag4", "month"]            # month = calendar month the complaint was received
RAW_INPUT = CATEGORICAL + NUMERIC                     # what the API and the pipelines accept
DERIVED = ["log_company_volume_lag4", "company_volume_lag4_missing", "subissue_unknown"]
FEATURES = CATEGORICAL + NUMERIC + DERIVED
# Fields that must never reach the model (leakage or protected): documented and asserted in tests.
FORBIDDEN = {"Company response to consumer", "Timely response?", "Tags", "tag_group", "tag_older", "tag_servicemember",
             "tag_recorded", "Complaint ID", "State", "Date received", "monetary_relief", "untimely", "week_start", "year"}

# Time-aware split on received week. Open cases are right-censored, so the window ends 2026-06-30
# (four months of closure maturity before the data snapshot).
WINDOW_END = "2026-06-30"
TRAIN_END = "2025-03-31"          # 2020-01 .. 2025-03  (includes the Jan-2025 Cash App surge on purpose)
VALID_END = "2025-09-30"          # 2025-04 .. 2025-09  model and threshold selection
# TEST: 2025-10-01 .. 2026-06-30  touched once, after all choices are frozen

REVIEW_CAPACITY = 0.20            # triage can review the top 20% of complaints by risk

# Pre-registered success criteria
CRITERIA = {
    PRIMARY: {"roc_auc_min": 0.70, "pr_auc_lift_min": 2.0, "recall_at_capacity_min": 0.45},
    SECONDARY: {"pr_auc_lift_min": 3.0},
    "fairness": {"disparate_impact_min": 0.80, "equalized_odds_diff_max": 0.10, "demographic_parity_diff_max": 0.10},
}
PROTECTED = {"older_american": "tag_older", "servicemember": "tag_servicemember"}
