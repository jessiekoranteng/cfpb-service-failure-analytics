"""Dashboard tests: each page renders without error, the what-if page agrees with the API scorer, and the Fairlearn-on-aggregates
engine reproduces the Module 4 numbers."""
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from src.dashboard import fair

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "app" / "assets" / "dashboard_data.json").read_text())
PAGES = ["Overview", "What drives predictions", "Try a complaint (what-if)", "Ethical Compliance", "Limits and transparency"]


def test_every_page_renders_without_exception():
    for p in ["overview", "drivers", "whatif", "ethics", "limits"]:
        at = AppTest.from_file(str(ROOT / "app" / "views" / f"{p}.py"), default_timeout=90)
        at.run()
        assert not at.exception, f"{p}: {at.exception}"


def test_full_app_navigation_runs():
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    assert any("Which complaints deserve" in t.value for t in at.title)


def test_whatif_matches_api_scorer():
    """The page and POST /predict share one scorer, so identical inputs must give identical numbers."""
    from fastapi.testclient import TestClient

    from api.main import app

    at = AppTest.from_file(str(ROOT / "app" / "views" / "whatif.py"), default_timeout=90)
    at.run()
    assert not at.exception
    payload = {"product": at.session_state["a_product"], "sub_product": at.session_state["a_sub_product"], "issue": at.session_state["a_issue"],
               "sub_issue": at.session_state["a_sub_issue"], "channel": at.session_state["a_channel"], "region": at.session_state["a_region"],
               "company": at.session_state["a_company"], "month": at.session_state["a_month"], "company_volume_lag4": at.session_state["a_volume"]}
    with TestClient(app) as client:
        api = client.post("/predict", json=payload).json()
    shown = [m.value for m in at.metric if m.label == "Relief chance"]
    assert shown[0] == f"{100 * api['relief_probability']:.1f}%"


def test_changing_a_field_in_scenario_b_changes_only_b():
    at = AppTest.from_file(str(ROOT / "app" / "views" / "whatif.py"), default_timeout=90)
    at.run()
    at.session_state["b_sub_issue"] = "Deposits and withdrawals"
    at.run()
    assert not at.exception
    a, b = [m.value for m in at.metric if m.label == "Relief chance"][:2]
    assert a != b


def test_preset_buttons_load_the_worked_examples():
    at = AppTest.from_file(str(ROOT / "app" / "views" / "whatif.py"), default_timeout=90)
    at.run()
    next(b for b in at.button if b.label.startswith("A ·")).click().run()
    assert not at.exception
    assert at.session_state["a_company"] == DATA["examples"][0]["inputs"]["company"]
    score = next(m.value for m in at.metric if m.label == "Relief chance")
    assert score == f"{100 * DATA['examples'][0]['score']:.1f}%"


def test_dependent_dropdowns_never_hold_an_invalid_combination():
    at = AppTest.from_file(str(ROOT / "app" / "views" / "whatif.py"), default_timeout=90)
    at.run()
    at.session_state["a_product"] = "Money transfer, virtual currency, or money service"
    at.run()
    assert not at.exception
    assert at.session_state["a_issue"] in DATA["vocab"]["issue"][at.session_state["a_product"]]
    assert at.session_state["a_sub_issue"] in DATA["vocab"]["sub_issue"][f"{at.session_state['a_product']}||{at.session_state['a_issue']}"]


def test_histogram_threshold_is_exact_at_the_global_line():
    meta, F = DATA["meta"], DATA["fairness"]
    for dim in F["hist"]:
        assert fair.counts_from_hist(F["hist"][dim], meta["global_threshold"], meta["bin_width"]) == F["global_counts"][dim]


def test_fairlearn_on_aggregates_reproduces_module4_region_numbers():
    F = DATA["fairness"]
    before = fair.summarise(F["global_counts"]["region"])
    after = fair.summarise(F["deployed_counts"]["region"])
    rb, ra = F["region_before"], F["region_after"]
    assert before["tpr_gap"] == pytest.approx(rb["tpr_gap"], abs=2e-4)
    assert before["fpr_gap"] == pytest.approx(rb["fpr_gap"], abs=2e-4)
    assert before["disparate_impact_ratio"] == pytest.approx(rb["selection_ratio_min_max"], abs=1e-3)
    assert after["tpr_gap"] == pytest.approx(ra["tpr_gap"], abs=2e-4)
    assert after["fpr_gap"] == pytest.approx(ra["fpr_gap"], abs=2e-4)
    assert after["flagged_share"] == pytest.approx(ra["flag_share"], abs=1e-4)


def test_fairlearn_on_aggregates_equals_fairlearn_on_expanded_rows():
    """Weighted aggregates must give the same answer as the explicit one-row-per-complaint table."""
    import numpy as np
    from fairlearn.metrics import MetricFrame, true_positive_rate

    counts = DATA["fairness"]["deployed_counts"]["tag_older"]
    y, p, g = [], [], []
    for grp, c in counts.items():
        for cell, yt, yp in fair.CELLS:
            y += [yt] * c[cell]; p += [yp] * c[cell]; g += [grp] * c[cell]
    mf = MetricFrame(metrics=true_positive_rate, y_true=np.array(y), y_pred=np.array(p), sensitive_features=np.array(g))
    t = fair.summarise(counts)["table"].set_index("Group")["tpr"]
    for grp in counts:
        assert t[grp] == pytest.approx(mf.by_group[grp], abs=1e-12)


def test_slider_moves_the_metrics_in_the_expected_direction():
    meta, F = DATA["meta"], DATA["fairness"]
    lo = fair.summarise(fair.counts_from_hist(F["hist"]["region"], 0.10, meta["bin_width"]))
    hi = fair.summarise(fair.counts_from_hist(F["hist"]["region"], 0.30, meta["bin_width"]))
    assert lo["flagged_share"] > hi["flagged_share"]


def test_dashboard_data_contains_no_complaint_level_fields():
    banned = {"complaint_key", "Complaint ID", "narrative", "Consumer complaint narrative", "ZIP code", "State", "Date received"}
    text = json.dumps(DATA)
    for b in banned:
        assert b not in text
    assert all(len(v) <= 400 for v in (h["neg"] for h in DATA["fairness"]["hist"]["region"].values()))


def test_example_scores_match_the_scorer():
    from src.modeling.scoring import Scorer

    sc = Scorer.from_dir(explain=False)
    for e in DATA["examples"]:
        r = sc.score({**e["inputs"]}, reasons=False)
        assert r["relief_probability"] == pytest.approx(e["score"], abs=1e-4)
        assert r["flag_for_review"] == e["flagged_regional"]


def test_no_page_makes_a_recommendation_or_decision_claim():
    text = " ".join(p.read_text().lower() for p in (ROOT / "app" / "views").glob("*.py"))
    for phrase in ("we recommend", "you should approve", "deny the complaint", "executive summary"):
        assert phrase not in text


def test_goal_scorecard_covers_g1_to_g5_and_reports_misses():
    g = {x["id"]: x for x in DATA["goals"]}
    assert list(g) == ["G1", "G2", "G3", "G4", "G5"]
    assert g["G1"]["status"] == "Partly met" and g["G5"]["status"] == "Not assessed"
    assert "66%" in g["G2"]["observed"]
