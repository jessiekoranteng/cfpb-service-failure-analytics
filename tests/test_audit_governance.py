import json
import os
import time

import pandas as pd
import pytest

from src.pipeline.audit import AuditLog, frame_fingerprint, verify_chain
from src.pipeline.governance import check_access, enforce_retention, load_policy

POLICY = load_policy(os.path.join(os.path.dirname(__file__), "..", "governance", "policy.yaml"))


def test_audit_events_are_chained_and_value_free(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl", who="tester")
    df = pd.DataFrame({"x": ["secret-value-1", "secret-value-2"]})

    @log.audited("transform", "demo", "unit test")
    def f(d):
        return d.assign(y=1)

    f(df)
    f(df)
    ok, n = verify_chain(tmp_path / "a.jsonl")
    assert ok and n == 2
    text = (tmp_path / "a.jsonl").read_text()
    assert "secret-value" not in text and "tester" in text
    rec = json.loads(text.splitlines()[0])
    assert rec["rows_in"] == 2 and rec["columns"] == ["x", "y"] and len(rec["output_sha256"]) == 64


def test_tampering_is_detected(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl", who="t")
    for i in range(3):
        log.event("read", f"d{i}", "test")
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    lines[1] = lines[1].replace('"d1"', '"dX"')
    (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n")
    assert verify_chain(tmp_path / "a.jsonl")[0] is False
    del lines[1]
    (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n")
    assert verify_chain(tmp_path / "a.jsonl")[0] is False


def test_fingerprint_changes_with_content():
    a = pd.DataFrame({"x": [1, 2]})
    assert frame_fingerprint(a) == frame_fingerprint(a.copy())
    assert frame_fingerprint(a) != frame_fingerprint(pd.DataFrame({"x": [1, 3]}))


def test_access_control_allows_and_denies_and_logs(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl", who="t")
    assert check_access(POLICY, "data_engineer", "raw", "write", log)
    assert check_access(POLICY, "ethics_panel", "processed_audit", "read", log)
    for role, layer, action in [("analyst", "processed_audit", "read"), ("viewer", "raw", "read"),
                                ("analyst", "processed_analysis", "write"), ("auditor", "audit_log", "append"),
                                ("intruder", "raw", "read")]:
        with pytest.raises(PermissionError):
            check_access(POLICY, role, layer, action, log)
    outcomes = [json.loads(line)["outcome"] for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert outcomes.count("allowed") == 2 and outcomes.count("denied") == 5
    assert verify_chain(tmp_path / "a.jsonl")[0]


def test_retention_dry_run_then_delete(tmp_path):
    interim = tmp_path / "data" / "interim"
    interim.mkdir(parents=True)
    old, new = interim / "old.parquet", interim / "new.parquet"
    old.write_text("x"); new.write_text("y")
    now = time.time()
    os.utime(old, (now - 200 * 86400, now - 200 * 86400))     # beyond the 90-day window
    log = AuditLog(tmp_path / "a.jsonl", who="t")
    assert enforce_retention(POLICY, "interim", tmp_path, now, dry_run=True, log=log) == ["data/interim/old.parquet"]
    assert old.exists()
    enforce_retention(POLICY, "interim", tmp_path, now, dry_run=False, log=log)
    assert not old.exists() and new.exists()
