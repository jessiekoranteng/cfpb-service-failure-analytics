"""Retraining decision rules, gates and the promotion guard."""
from datetime import UTC, datetime

import joblib
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.modeling import config as C
from src.modeling import retrain as R
from tests.test_modeling import synthetic, tiny  # noqa: F401  (fixture re-used)


def mon(levels_by_month=None, drifted=0.0, breach=False, alerts=()):
    months = [{"month": f"2026-0{i + 1}", "levels": {"roc_auc": "ok", "pr_auc_lift": "ok", "recall_at_flag": "ok", "ece": "ok", "psi_company": "ok",
                                                      **(levels_by_month[i] if levels_by_month else {})}} for i in range(4)]
    return {"monthly": months, "alerts": list(alerts),
            "drift": {"drifted_columns": {"share": drifted, "count": drifted * 7}, "columns": [0] * 7},
            "fairness_tags": {"replay_window": {"older_american": {"confirmed_breach": breach}, "servicemember": {"confirmed_breach": False}}}}


NOW = datetime(2026, 11, 20, tzinfo=UTC)


def test_quiet_month_triggers_nothing():
    r = R.decide(mon(), trained_at="2026-11-01T00:00:00+00:00", today=NOW)
    assert r["decision"] == "NO RETRAIN TRIGGERED" and not r["reasons"]


def test_hard_performance_breach_recommends_retraining():
    r = R.decide(mon([{}, {"roc_auc": "hard"}, {}, {}]), "2026-11-01T00:00:00+00:00", NOW)
    assert r["decision"] == "RETRAIN RECOMMENDED" and "roc_auc" in r["reasons"][0]


def test_single_warning_is_watched_but_two_in_a_row_trigger():
    one = R.decide(mon([{}, {"ece": "warn"}, {}, {}]), "2026-11-01T00:00:00+00:00", NOW)
    two = R.decide(mon([{}, {"ece": "warn"}, {"ece": "warn"}, {}]), "2026-11-01T00:00:00+00:00", NOW)
    assert one["decision"] == "NO RETRAIN TRIGGERED" and two["decision"] == "RETRAIN RECOMMENDED"


def test_company_mix_drift_needs_persistence_and_half_the_columns_trigger():
    assert R.decide(mon([{"psi_company": "hard"}, {}, {}, {}]), None, NOW)["decision"] == "NO RETRAIN TRIGGERED"
    assert R.decide(mon([{}, {}, {"psi_company": "hard"}, {"psi_company": "hard"}]), None, NOW)["decision"] == "RETRAIN RECOMMENDED"
    assert R.decide(mon(drifted=0.6), None, NOW)["decision"] == "RETRAIN RECOMMENDED"


def test_a_confirmed_fairness_breach_escalates_and_is_never_a_silent_retrain():
    r = R.decide(mon([{"roc_auc": "hard"}, {}, {}, {}], breach=True), None, NOW)
    assert r["decision"].startswith("ESCALATE") and r["escalations"]


def test_scheduled_retrain_after_six_months():
    r = R.decide(mon(), "2026-01-01T00:00:00+00:00", NOW)
    assert r["decision"] == "RETRAIN RECOMMENDED" and "scheduled" in r["reasons"][0]


def metrics(**kw):
    base = {"roc_auc": 0.79, "pr_auc": 0.31, "ece": 0.015, "flag_share": 0.20, "region_selection_ratio": 0.85,
            "f2_older_american": {"tpr_gap": 0.04, "fpr_gap": 0.03}, "f2_servicemember": {"tpr_gap": 0.02, "fpr_gap": 0.02}}
    return {**base, **kw}


def failing(rep):
    return {g["gate"].split(":")[0] + ":" + g["gate"].split(":")[1][:22] for g in rep["gates"] if not g["passed"]}


def test_equal_challenger_passes_every_gate():
    rep = R.run_gates(metrics(), metrics(), list(C.RAW_INPUT), 0.95)
    assert rep["passed"] and all(g["passed"] for g in rep["gates"])


@pytest.mark.parametrize("change,expect", [
    ({"roc_auc": 0.69}, "ROC-AUC"), ({"roc_auc": 0.77}, "ROC-AUC"), ({"pr_auc": 0.29}, "PR-AUC"), ({"ece": 0.06}, "ECE"),
    ({"flag_share": 0.30}, "flagged share"), ({"region_selection_ratio": 0.75}, "selection ratio"),
    ({"f2_older_american": {"tpr_gap": 0.09, "fpr_gap": 0.03}}, "older_american tpr_gap"),
])
def test_each_gate_blocks_on_its_own(change, expect):
    rep = R.run_gates(metrics(), metrics(**change), list(C.RAW_INPUT), 0.95)
    assert not rep["passed"]
    assert any(expect in g["gate"] and not g["passed"] for g in rep["gates"])


def test_fairness_gap_may_be_worse_than_champion_if_still_within_the_charter_limit():
    rep = R.run_gates(metrics(f2_servicemember={"tpr_gap": 0.01, "fpr_gap": 0.01}), metrics(f2_servicemember={"tpr_gap": 0.045, "fpr_gap": 0.01}), list(C.RAW_INPUT), 0.95)
    assert rep["passed"]


def test_unstable_explanations_block_the_challenger():
    rep = R.run_gates(metrics(), metrics(), list(C.RAW_INPUT), 0.55)
    assert not rep["passed"] and any("explanation stability" in g["gate"] and not g["passed"] for g in rep["gates"])


def test_explanation_stability_is_one_for_identical_models(champion):
    d = synthetic()
    assert R.explanation_stability(champion, champion, d[C.RAW_INPUT], n=300) == pytest.approx(1.0)


def test_protected_or_changed_inputs_fail_the_contract_gates():
    rep = R.run_gates(metrics(), metrics(), list(C.RAW_INPUT) + ["tag_older"], 0.95)
    assert not rep["passed"]
    assert {"no forbidden or protected input", "input contract unchanged"} <= {g["gate"] for g in rep["gates"] if not g["passed"]}


@pytest.fixture()
def champion(tiny, tmp_path):  # noqa: F811
    d, pipe = tiny
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic").fit(d[C.RAW_INPUT], d["y"])
    b = {"calibrated": cal, "base": pipe, "target": "y", "features": C.RAW_INPUT, "algorithm": "lightgbm", "review_threshold": 0.3,
         "trained_at": "2026-01-01T00:00:00+00:00"}
    return b


def test_challenger_clones_the_champion_algorithm_and_never_touches_it(champion):
    d = synthetic()
    d["week_start"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(range(len(d)), unit="h") * 5
    cut = d["week_start"].quantile([0.6, 0.8, 1.0]).dt.strftime("%Y-%m-%d").tolist()
    windows = {"fit_end": cut[0], "calib_end": cut[1], "gate_end": cut[2]}
    frozen = champion["calibrated"].predict_proba(d[C.RAW_INPUT].head(50))[:, 1].copy()
    ch = R.build_challenger(d, champion, windows, target="y")
    assert ch["algorithm"] == champion["algorithm"] and ch["base"] is not champion["base"]
    assert ch["base"][-1].get_params()["num_leaves"] == champion["base"][-1].get_params()["num_leaves"]
    assert ch["n_fit"] > 0 and ch["n_calib"] > 0 and set(ch["region_thresholds"]) <= set(d["region"])
    assert champion["review_threshold"] == 0.3
    assert (champion["calibrated"].predict_proba(d[C.RAW_INPUT].head(50))[:, 1] == frozen).all()   # champion untouched


def registry(tmp_path, champion, dry_run, passed):
    mlflow = R._mlflow(f"sqlite:///{tmp_path}/m.db")
    from mlflow.tracking import MlflowClient
    with mlflow.start_run():
        mlflow.sklearn.log_model(champion["calibrated"], name="model", serialization_format="cloudpickle", registered_model_name=R.MODEL_NAME)
    MlflowClient().set_registered_model_alias(R.MODEL_NAME, "champion", "1")
    rep = {"passed": passed, "champion": {"roc_auc": 0.79}, "challenger": {"roc_auc": 0.80}}
    b = dict(champion, windows={"fit_end": "a", "calib_end": "b", "gate_end": "c"}, n_fit=1, n_calib=1)
    return R.register_challenger(b, rep, f"sqlite:///{tmp_path}/m.db", dry_run, tmp_path / "models")


def test_promotion_guards(tmp_path, champion):
    (tmp_path / "models").mkdir()
    uri = f"sqlite:///{tmp_path}/m.db"
    v_dry = registry(tmp_path, champion, dry_run=True, passed=True)
    with pytest.raises(PermissionError, match="dry run"):
        R.promote(v_dry, "A. Mensah, Panel chair", "T-1", uri, tmp_path / "models")
    v_fail = registry(tmp_path, champion, dry_run=False, passed=False)
    with pytest.raises(PermissionError, match="every gate"):
        R.promote(v_fail, "A. Mensah, Panel chair", "T-1", uri, tmp_path / "models")
    v_ok = registry(tmp_path, champion, dry_run=False, passed=True)
    with pytest.raises(PermissionError, match="approver"):
        R.promote(v_ok, "  ", "T-1", uri, tmp_path / "models")
    with pytest.raises(PermissionError, match="ticket"):
        R.promote(v_ok, "A. Mensah", "", uri, tmp_path / "models")


def test_successful_promotion_archives_the_old_bundle_and_moves_the_alias(tmp_path, champion):
    models = tmp_path / "models"
    models.mkdir()
    joblib.dump(dict(champion, review_threshold=0.111), models / f"risk_{C.PRIMARY}.joblib")
    uri = f"sqlite:///{tmp_path}/m.db"
    v = registry(tmp_path, dict(champion, review_threshold=0.222), dry_run=False, passed=True)
    out = R.promote(v, "A. Mensah, Panel chair", "T-42", uri, models)
    assert out["promoted"] == v and out["replaced"] == "1"
    assert joblib.load(models / f"risk_{C.PRIMARY}.joblib")["review_threshold"] == 0.222
    archived = list((models / "archive").glob("*.joblib"))
    assert len(archived) == 1 and joblib.load(archived[0])["review_threshold"] == 0.111
    from mlflow.tracking import MlflowClient
    c = MlflowClient()
    assert str(c.get_model_version_by_alias(R.MODEL_NAME, "champion").version) == str(v)
    assert c.get_model_version(R.MODEL_NAME, str(v)).tags["promoted_by"] == "A. Mensah, Panel chair"


def test_challenger_registration_leaves_the_champion_alias_alone(tmp_path, champion):
    (tmp_path / "models").mkdir()
    v = registry(tmp_path, champion, dry_run=True, passed=False)
    from mlflow.tracking import MlflowClient
    c = MlflowClient()
    assert str(c.get_model_version_by_alias(R.MODEL_NAME, "champion").version) == "1"
    assert str(c.get_model_version_by_alias(R.MODEL_NAME, "challenger").version) == str(v)
