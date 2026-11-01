import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.modeling import charter as CH
from src.modeling import config as C
from src.modeling import data as D
from src.modeling import fairness as F
from src.modeling import metrics as M
from src.modeling import pipeline as P
from src.modeling import sensitivity as S


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    issue = rng.choice(["Fraud or scam", "Managing an account", "Other transaction problem"], n, p=[0.3, 0.4, 0.3])
    d = pd.DataFrame({
        "Product": rng.choice(["Checking or savings account", "Money transfer, virtual currency, or money service"], n),
        "Sub-product": rng.choice(["Checking account", "Domestic (US) money transfer", "Unknown"], n),
        "Issue": issue, "Sub-issue": rng.choice(["Unknown", "Fee problem", "Account frozen"], n),
        "Submitted via": rng.choice(["Web", "Phone", "Referral"], n), "region": rng.choice(["South", "West", "Northeast"], n),
        "company_key": rng.choice(["ACME BANK", "BETA PAY", "GAMMA CU"], n),
        "company_volume_lag4": rng.integers(0, 800, n).astype("float32"), "month": rng.integers(1, 13, n),
    })
    p = 0.05 + 0.4 * (d["Issue"] == "Fraud or scam") + 0.2 * (d["company_key"] == "ACME BANK")
    d["y"] = (rng.random(n) < p.clip(0, 0.9)).astype(int)
    d["week_start"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
    return d


@pytest.fixture(scope="module")
def tiny():
    d = synthetic()
    pipe = P.make_lgbm(n_estimators=40, num_leaves=7, min_child_samples=20, min_data_per_group=20, n_jobs=1).fit(d[C.RAW_INPUT], d["y"])
    return d, pipe


def test_feature_builder_derives_known_values():
    d = pd.DataFrame([{"Product": "p", "Sub-product": "s", "Issue": "i", "Sub-issue": "Unknown", "Submitted via": "Web", "region": "South",
                       "company_key": "ACME", "company_volume_lag4": np.nan, "month": 3}])
    out = P.FeatureBuilder().transform(d)
    assert list(out.columns) == C.FEATURES
    assert out["company_volume_lag4_missing"].iloc[0] == 1 and out["subissue_unknown"].iloc[0] == 1
    assert out["log_company_volume_lag4"].iloc[0] == 0


def test_forbidden_columns_are_rejected():
    with pytest.raises(ValueError, match="forbidden"):
        D.assert_no_leakage(C.RAW_INPUT + ["tag_older"])
    D.assert_no_leakage(C.RAW_INPUT)  # the real feature set is clean


def test_split_is_time_ordered_disjoint_and_drops_open_cases():
    d = synthetic(n=500)
    d["week_start"] = pd.date_range("2020-01-06", "2026-06-29", periods=500).normalize()
    d["monetary_relief"] = d["y"].astype("float")
    d.loc[d.index[:5], "monetary_relief"] = np.nan
    d.loc[d.index[-3:], "week_start"] = pd.Timestamp("2027-01-04")  # after the observation window
    sp = D.split(d, "monetary_relief")
    assert len(sp["train"]) and len(sp["valid"]) and len(sp["test"])
    assert sp["train"]["week_start"].max() < sp["valid"]["week_start"].min() <= sp["valid"]["week_start"].max() < sp["test"]["week_start"].min()
    assert sum(len(v) for v in sp.values()) == 500 - 5 - 3
    assert all(v["monetary_relief"].notna().all() for v in sp.values())


def test_metrics_perfect_and_random():
    y = np.array([0] * 90 + [1] * 10)
    perfect = M.evaluate(y, y.astype(float), 0.5)
    assert perfect["roc_auc"] == 1 and perfect["recall"] == 1 and perfect["recall_at_capacity"] == 1
    rnd = M.evaluate(y, np.random.default_rng(0).random(100), 0.5)
    assert 0.2 < rnd["roc_auc"] < 0.8
    assert M.top_share_threshold(np.arange(100.0), 0.2) == pytest.approx(79.2)


def test_pipeline_learns_signal_and_handles_unseen_category(tiny):
    d, pipe = tiny
    s = pipe.predict_proba(d[C.RAW_INPUT])[:, 1]
    assert M.evaluate(d["y"], s, 0.3)["roc_auc"] > 0.65
    new = d[C.RAW_INPUT].head(2).assign(company_key="NEVER SEEN", Issue="Brand new issue")
    assert np.isfinite(pipe.predict_proba(new)[:, 1]).all()


def test_fairness_report_detects_a_planted_gap():
    rng = np.random.default_rng(1)
    n = 4000
    flag = rng.integers(0, 2, n)
    y = rng.integers(0, 2, n)
    pred = (rng.random(n) < 0.15 + 0.2 * flag).astype(int)
    g = pd.DataFrame({"tag_older": flag, "tag_servicemember": 1 - flag, "tag_group": np.where(flag == 1, "Older American", "None recorded"),
                      "region": rng.choice(["South", "West"], n)})
    rep = F.audit(y, pred, g)["older_american"]
    assert rep["selection_group"] > rep["selection_rest"] + 0.1
    assert rep["disparate_impact_ratio"] < 0.8 and not rep["passes"]["disparate_impact"]
    fair = F.audit(y, (rng.random(n) < 0.2).astype(int), g)["older_american"]
    assert fair["passes"]["demographic_parity"]


def test_flag_top_flags_requested_share():
    s = np.random.default_rng(0).random(1000)
    assert F.flag_top(s, 0.2).mean() == pytest.approx(0.2, abs=0.01)


def test_sensitivity_table_has_all_scenarios_and_small_noop_change(tiny):
    d, pipe = tiny
    t = S.perturbation_table(pipe, d[C.RAW_INPUT], d["y"], thr=0.3)
    assert t.iloc[0]["scenario"] == "baseline" and len(t) == 8
    assert (t["flag_flip_rate"].between(0, 1)).all()
    ci = S.bootstrap_ci(d["y"], pipe.predict_proba(d[C.RAW_INPUT])[:, 1], n_boot=50)
    assert ci["roc_auc_ci"][0] < ci["roc_auc_ci"][1]


def test_surge_mask_selects_three_january_weeks():
    d = pd.DataFrame({"week_start": pd.to_datetime(["2025-01-06", "2025-01-13", "2025-01-27", "2025-02-03"])})
    assert S.surge_mask(d).tolist() == [False, True, True, False]


@pytest.fixture()
def api_client(tiny, tmp_path, monkeypatch):
    import joblib
    from fastapi.testclient import TestClient
    d, pipe = tiny
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic").fit(d[C.RAW_INPUT], d["y"])
    for t in C.TARGETS:
        joblib.dump({"calibrated": cal, "base": pipe, "target": t, "features": C.RAW_INPUT, "algorithm": "lightgbm",
                     "review_threshold": 0.3, "trained_at": "2026-01-01T00:00:00+00:00"}, tmp_path / f"risk_{t}.joblib")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    from api.main import app
    with TestClient(app) as c:
        yield c


GOOD = {"product": "Checking or savings account", "sub_product": "Checking account", "issue": "Fraud or scam", "sub_issue": "Fee problem",
        "channel": "Web", "region": "South", "company": "acme bank", "company_volume_lag4": 300, "month": 5}


def test_api_predict_returns_probabilities_reasons_and_flag(api_client):
    r = api_client.post("/predict", json=GOOD)
    assert r.status_code == 200
    b = r.json()
    assert 0 <= b["relief_probability"] <= 1 and 0 <= b["untimely_probability"] <= 1
    assert isinstance(b["flag_for_review"], bool) and len(b["reasons"]) == 3 and b["unseen_values"] == []
    assert "decision support" in b["note"].lower()


def test_api_flags_unseen_values_and_accepts_missing_volume(api_client):
    r = api_client.post("/predict", json={**GOOD, "company": "Brand New Bank", "company_volume_lag4": None})
    assert r.status_code == 200 and any(u.startswith("company_key=") for u in r.json()["unseen_values"])
    # reasons must show what the caller sent, never the internal code ("nan")
    assert all("nan" not in x["value"].lower() for x in r.json()["reasons"])
    for x in r.json()["reasons"]:
        if x["feature"] == "Company":
            assert "BRAND NEW BANK" in x["value"] and "not seen in training" in x["value"]


@pytest.mark.parametrize("bad", [{"month": 13}, {"month": 0}, {"company_volume_lag4": -1}, {"issue": ""}, {"channel": "x" * 100}])
def test_api_rejects_invalid_input(api_client, bad):
    assert api_client.post("/predict", json={**GOOD, **bad}).status_code == 422


def test_api_rejects_protected_or_extra_fields_gracefully(api_client):
    r = api_client.post("/predict", json={**GOOD, "older_american": 1})
    assert r.status_code == 200 and "older_american" not in r.text


def test_api_health_and_model_info(api_client):
    assert api_client.get("/health").json()["status"] == "ok"
    info = api_client.get("/model-info").json()
    assert info["features"] == C.RAW_INPUT and info["relief"]["algorithm"] == "lightgbm"


def test_api_uses_region_specific_threshold_when_present(tiny, tmp_path, monkeypatch):
    import joblib
    from fastapi.testclient import TestClient
    d, pipe = tiny
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic").fit(d[C.RAW_INPUT], d["y"])
    for t in C.TARGETS:
        joblib.dump({"calibrated": cal, "base": pipe, "target": t, "features": C.RAW_INPUT, "algorithm": "lightgbm", "review_threshold": 0.3,
                     "region_thresholds": {"South": 0.01, "West": 0.99}, "trained_at": "x"}, tmp_path / f"risk_{t}.joblib")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    from api.main import app
    with TestClient(app) as c:
        south = c.post("/predict", json={**GOOD, "region": "South"}).json()
        west = c.post("/predict", json={**GOOD, "region": "West"}).json()
        other = c.post("/predict", json={**GOOD, "region": "Other"}).json()
    assert south["flag_for_review"] is True and south["review_threshold"] == 0.01
    assert west["flag_for_review"] is False and west["review_threshold"] == 0.99
    assert other["review_threshold"] == 0.3


# ---- Ethics Charter objectives (F2 gaps, F4 no decisions) -------------------------------------------------------------

def _planted(n=6000, seed=1, bias=0.0):
    """Outcomes y, flags that depend on y (a real signal) and, if bias > 0, extra flagging of the 'Older American' group."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    grp = rng.integers(0, 3, n)
    older, service = (grp == 1).astype(int), (grp == 2).astype(int)
    flags = (rng.random(n) < 0.15 + 0.35 * y + bias * older).astype(int)
    g = pd.DataFrame({"tag_group": np.array(["None recorded", "Older American", "Servicemember"])[grp], "region": "South",
                      "tag_older": older, "tag_servicemember": service})
    return y, flags, g


def test_charter_gaps_agree_with_fairlearn():
    y, flags, g = _planted()
    res, rep = CH.f2_check(y, flags, g, n_boot=20)
    # F.audit rounds its group tables to 4 decimals; the bincount path used for bootstrapping is exact
    assert res["four_group"]["tpr_gap"]["gap"] == pytest.approx(rep["tag_group"]["tpr_gap"], abs=1e-4)
    assert res["four_group"]["fpr_gap"]["gap"] == pytest.approx(rep["tag_group"]["fpr_gap"], abs=1e-4)
    o = rep["older_american"]
    assert res["older_american"]["tpr_gap"]["gap"] == pytest.approx(abs(o["tpr_group"] - o["tpr_rest"]), abs=1e-9)
    lo, hi = res["older_american"]["fpr_gap"]["ci95"]
    assert lo <= hi


def test_f2_holds_for_a_group_blind_rule_and_fails_for_a_planted_gap():
    y, flags, g = _planted(bias=0.0)
    fair, _ = CH.f2_check(y, flags, g, n_boot=20)
    assert fair["older_american"]["holds"]
    y, flags, g = _planted(bias=0.30)
    unfair, _ = CH.f2_check(y, flags, g, n_boot=20)
    assert not unfair["older_american"]["holds"] and unfair["older_american"]["fpr_gap"]["gap"] > CH.F2_LIMIT


def test_deployed_flags_use_the_region_threshold_and_fall_back_to_the_global_one():
    bundle = {"review_threshold": 0.5, "region_thresholds": {"West": 0.2}}
    df = pd.DataFrame({"region": ["West", "South", "Nowhere"]})
    assert CH.deployed_flags(bundle, df, [0.3, 0.3, 0.6]).tolist() == [1, 0, 1]


def test_api_never_issues_a_decision_or_touches_a_deadline(api_client):
    """Charter F4: the service only reorders human attention. Its response has a flag and reasons, never a decision."""
    body = api_client.post("/predict", json=GOOD).json()
    assert not set(body) & {"decision", "close", "deadline", "deprioritise", "priority"}
    assert isinstance(body["flag_for_review"], bool) and "a person decides" in body["note"]
