"""Monitoring statistics, alert rules and the job's output contract."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring import job as J
from src.monitoring import metrics as M
from src.monitoring import thresholds as T

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)


def test_psi_is_zero_for_identical_and_large_for_shifted_distributions():
    a = pd.Series(rng.choice(list("abcde"), 5000, p=[.4, .3, .15, .1, .05]))
    assert M.psi_categorical(a, a) == pytest.approx(0, abs=1e-9)
    b = pd.Series(rng.choice(list("abcde"), 5000, p=[.05, .1, .15, .3, .4]))
    assert M.psi_categorical(a, b) > 0.5
    s = rng.beta(2, 12, 5000)
    assert M.psi_numeric(s, s) == pytest.approx(0, abs=1e-9)
    assert M.psi_numeric(s, rng.beta(4, 8, 5000)) > 0.2


def test_psi_counts_unseen_categories_as_other_not_as_an_error():
    ref, cur = pd.Series(["a"] * 90 + ["b"] * 10), pd.Series(["a"] * 50 + ["zzz"] * 50)
    assert M.psi_categorical(ref, cur) > 0.2


def test_ece_is_zero_for_perfectly_calibrated_scores_and_positive_otherwise():
    p = rng.uniform(0, 1, 40000)
    y = (rng.uniform(0, 1, 40000) < p).astype(int)
    assert M.ece(y, p) < 0.02
    assert M.ece(y, np.clip(p + 0.2, 0, 1)) > 0.1


def test_group_rates_suppress_small_cells():
    y = np.array([1] * 5 + [0] * 100)
    f = np.ones(105, int)
    r = M.group_rates(y, f, np.ones(105, bool))
    assert r["tpr"] is None and r["fpr"] == 1.0     # 5 positives < MIN_CELL, 100 negatives are enough


def test_vs_rest_gap_and_bootstrap_interval_contain_the_truth():
    n = 20000
    g = rng.uniform(size=n) < 0.2
    y = (rng.uniform(size=n) < 0.2).astype(int)
    flag = np.where(y == 1, rng.uniform(size=n) < np.where(g, 0.40, 0.55), rng.uniform(size=n) < 0.1).astype(int)
    gap = M.vs_rest_gaps(y, flag, g)
    assert gap["tpr_gap"] == pytest.approx(0.15, abs=0.04)
    ci = M.bootstrap_gaps(y, flag, g, n_boot=100)["tpr_ci95"]
    assert ci[0] < 0.15 < ci[1]


def test_size_tiers_use_training_counts_and_mark_unseen():
    train = pd.Series(["A"] * 100 + ["B"] * 50 + [f"C{i}" for i in range(400)])
    t = M.size_tiers(train, pd.Series(["A", "B", "C399", "NEW"]))
    assert list(t) == ["Top 5", "Top 5", "Small (301+)", "Unseen"]


def test_selection_ratio():
    flag = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    region = np.array(["x", "x", "x", "x", "y", "y", "y", "y"])
    assert M.selection_ratio(flag, region) == pytest.approx(0.25 / 0.5)


@pytest.mark.parametrize("signal,value,expected", [
    ("roc_auc", 0.80, "ok"), ("roc_auc", 0.72, "warn"), ("roc_auc", 0.69, "hard"),
    ("ece", 0.02, "ok"), ("ece", 0.04, "warn"), ("ece", 0.06, "hard"),
    ("flag_share", 0.20, "ok"), ("flag_share", 0.12, "warn"), ("flag_share", 0.35, "hard"),
    ("region_selection_ratio", 0.85, "ok"), ("region_selection_ratio", 0.75, "warn"), ("region_selection_ratio", 0.65, "hard"),
    ("psi_company", 0.30, "hard"), ("f2_tpr_gap", 0.06, "warn"), ("f2_tpr_gap", 0.05, "ok")])
def test_alert_levels(signal, value, expected):
    assert T.level(signal, value) == expected


def test_uncomputable_value_is_not_reported_as_ok():
    assert T.level("roc_auc", None) == "n/a" and T.level("ece", float("nan")) == "n/a"


def test_every_rule_has_a_documented_source_and_consistent_limits():
    for kind, warn, hard, src in T.RULES.values():
        assert src and kind in ("min", "max", "band")
        if kind == "min":
            assert warn > hard
        if kind == "max":
            assert warn < hard


def synthetic(n=12000, months=("2026-01", "2026-02", "2026-03")):
    p = rng.beta(2, 12, n)
    y = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame({"y": y, "relief_probability": p, "flag_for_review": p >= 0.17,
                       "month_received": rng.choice(months, n), "company_key": rng.choice(list("ABCDEFGH"), n),
                       "company_seen": rng.uniform(size=n) > 0.01, "region": rng.choice(["N", "S", "E"], n),
                       "Product": rng.choice(["P1", "P2"], n), "size_tier": rng.choice(["Top 5", "Small (301+)"], n),
                       "tag_older": rng.uniform(size=n) < 0.1, "tag_servicemember": rng.uniform(size=n) < 0.1,
                       "week_start": pd.to_datetime("2026-01-05") + pd.to_timedelta(rng.integers(0, 80, n), unit="D")})
    return df


def test_monthly_rows_carry_levels_and_alerts_only_for_breaches():
    ref, cur = synthetic(), synthetic()
    rows = J.monthly(ref, cur)
    assert [r["month"] for r in rows] == ["2026-01", "2026-02", "2026-03"]
    assert all(set(r["levels"]) >= {"roc_auc", "ece", "psi_company"} for r in rows)
    for a in J.alerts_from(rows):
        assert a["level"] in ("warn", "hard")


def test_thin_month_is_not_scored_for_performance():
    d = synthetic(300, months=("2026-01",))
    r = J.monthly(synthetic(), d)[0]
    assert r["roc_auc"] is None and r["levels"]["roc_auc"] == "n/a"


def test_fairness_cuts_suppress_thin_groups_and_agree_with_fairlearn():
    d = synthetic()
    d.loc[d.index[:30], "Product"] = "Tiny"
    d.loc[d["Product"] == "Tiny", "y"] = 0
    out = J.fairness_cuts(d)
    tiny = next(g for g in out["product"]["groups"] if g["group"] == "Tiny")
    assert tiny["suppressed"] and tiny["tpr"] is None
    assert {"product", "company_size_tier", "region"} == set(out)


def test_tag_fairness_has_quarterly_breakdown():
    out = J.fairness_tags(synthetic(), n_boot=20)
    assert set(out["replay_window"]) == {"older_american", "servicemember"}
    assert out["by_quarter"]


def test_evidently_report_flags_a_shifted_column(tmp_path):
    pytest.importorskip("evidently")
    ref, cur = synthetic(), synthetic()
    for d in (ref, cur):
        d["Issue"], d["Submitted via"] = "i", "Web"
        d["company_volume_lag4"] = rng.normal(500, 50, len(d))
    cur["company_volume_lag4"] += 400
    out = J.evidently_report(ref, cur, tmp_path / "r.html")
    by = {c["column"]: c for c in out["columns"]}
    assert by["company_volume_lag4"]["drifted"] and not by["region"]["drifted"]
    assert (tmp_path / "r.html").stat().st_size > 10000


ASSET = ROOT / "monitoring_app" / "assets" / "monitoring_data.json"


@pytest.mark.skipif(not ASSET.exists(), reason="monitoring_data.json not generated")
def test_committed_monitoring_asset_is_self_consistent_and_ties_to_module_4():
    d = json.loads(ASSET.read_text())
    assert d["replay"]["n"] == 94305 and d["reference"]["n"] == 50891
    for r in d["monthly"]:                                   # stored levels match the rules
        for k, lv in r["levels"].items():
            assert T.level(k, r[k]) == lv
    charter = json.loads((ROOT / "reports" / "m4" / "charter_checks.json").read_text())["deployed_region_thresholds"]["f2"]
    got = d["fairness_tags"]["replay_window"]["older_american"]
    assert got["tpr_gap"] == pytest.approx(charter["older_american"]["tpr_gap"]["gap"], abs=1e-6)
    assert got["fpr_gap"] == pytest.approx(charter["older_american"]["fpr_gap"]["gap"], abs=1e-6)
    assert d["crosscheck"]["replay_roc_auc"] == pytest.approx(0.791, abs=1e-3)      # Module 4 test ROC-AUC
    assert d["crosscheck"]["replay_ece"] == pytest.approx(0.0150, abs=5e-4)           # Module 4 test ECE
