"""The monitoring dashboard renders from the generated file and shows the numbers the job produced."""
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "monitoring_app" / "dashboard.py")
DATA = json.loads((ROOT / "monitoring_app" / "assets" / "monitoring_data.json").read_text())


def test_dashboard_renders_without_exception():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6


def test_headline_metrics_come_from_the_generated_file():
    at = AppTest.from_file(APP, default_timeout=90).run()
    m = {x.label: x.value for x in at.metric}
    assert m["ROC-AUC (replay)"] == f"{DATA['crosscheck']['replay_roc_auc']:.3f}"
    assert m["Calibration error"] == f"{DATA['crosscheck']['replay_ece']:.4f}"


def test_the_page_says_it_is_a_replay_and_not_live_traffic():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert any("replay, not live traffic" in w.value.lower() for w in at.warning)


def test_dashboard_data_holds_no_complaint_level_fields():
    text = json.dumps(DATA)
    for banned in ("complaint_key", "Complaint ID", "narrative", "ZIP"):
        assert banned not in text


def test_missing_data_file_stops_with_a_clear_error(tmp_path, monkeypatch):
    src = Path(APP).read_text().replace('HERE / "assets" / "monitoring_data.json"', f'Path(r"{tmp_path}/none.json")')
    f = tmp_path / "dash.py"
    f.write_text(src)
    at = AppTest.from_file(str(f), default_timeout=60).run()
    assert at.error and "monitoring_data.json is missing" in at.error[0].value
