import json

import pandas as pd
import pytest

from src.pipeline import validate as val
from src.pipeline.audit import verify_chain


def test_raw_suite_passes_on_real_sample(raw, tmp_path):
    s = val.run_suite(raw, "raw", tmp_path, min_rows=100, build_docs=False)
    assert s["success"] and s["failed"] == 0


def test_raw_suite_catches_bad_data(raw, tmp_path):
    bad = raw.copy()
    bad["Product"] = "Mortgage"
    s = val.run_suite(bad, "raw", tmp_path, min_rows=100, build_docs=False)
    assert not s["success"]
    with pytest.raises(ValueError, match="critical"):
        val.assert_no_critical_failures(s)


def test_curated_suite_passes(curated, tmp_path):
    s = val.run_suite(curated, "curated", tmp_path, min_rows=100, build_docs=False)
    assert s["success"], s


def test_flow_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "pf"))
    from flows.ccdb_pipeline import pipeline
    r = pipeline("tests/fixtures/ccdb_sample.csv", root=str(tmp_path), min_rows=100, min_group_block=1,
                 min_group_warn=10, feed_min_total=3)
    a = pd.read_parquet(tmp_path / "data/processed/analysis_table.parquet")
    assert r["audit_log"]["chain_ok"] and len(a) == r["rows"]["published"]
    banned = {"Complaint ID", "State", "Company", "Tags", "tag_group", "tag_older", "tag_servicemember", "Date received"}
    assert not banned & set(a.columns)
    assert pd.api.types.is_numeric_dtype(a["company_volume_lag4"]) and pd.api.types.is_datetime64_any_dtype(a["week_start"])
    assert json.loads((tmp_path / "reports/pipeline_run.json").read_text())["bias_overall"] in {"PASS", "WARN"}
    assert verify_chain(tmp_path / "logs/privacy_audit.jsonl")[0]
