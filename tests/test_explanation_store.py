"""Explanation store (audit trail) and API metrics."""
import json
from datetime import UTC, datetime, timedelta

import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.modeling import config as C
from src.modeling.explanation_store import GENESIS, ExplanationStore
from tests.test_modeling import GOOD, tiny  # noqa: F401  (fixture re-used)

RESULT = {"relief_probability": 0.31, "untimely_probability": 0.02, "flag_for_review": True, "review_threshold": 0.16,
          "reasons": [{"feature": "Company", "value": "ACME", "effect": 0.9}], "unseen_values": []}
INPUTS = {"product": "p", "issue": "i", "channel": "Web", "region": "South", "company": "ACME", "month": 5}
VERS = {"relief": "lightgbm@2026", "untimely": "lightgbm@2026"}


def test_store_round_trips_a_record_and_chains_hashes(tmp_path):
    s = ExplanationStore(tmp_path / "e.jsonl")
    a = s.append(INPUTS, RESULT, VERS)
    b = s.append(INPUTS, RESULT, VERS)
    ra, rb = s.get(a), s.get(b)
    assert ra["prev"] == GENESIS and rb["prev"] == ra["hash"]
    assert ra["reasons"] == RESULT["reasons"] and ra["model_versions"] == VERS
    assert s.get("nope") is None
    assert s.verify() == {"ok": True, "records": 2, "first_bad_index": None, "reason": None}


def test_editing_a_record_breaks_verification(tmp_path):
    s = ExplanationStore(tmp_path / "e.jsonl")
    for _ in range(3):
        s.append(INPUTS, RESULT, VERS)
    lines = (tmp_path / "e.jsonl").read_text().splitlines()
    rec = json.loads(lines[1])
    rec["relief_probability"] = 0.99          # someone rewrites the score after the fact
    lines[1] = json.dumps(rec)
    (tmp_path / "e.jsonl").write_text("\n".join(lines) + "\n")
    v = s.verify()
    assert v["ok"] is False and v["first_bad_index"] == 1


def test_deleting_a_record_breaks_the_chain(tmp_path):
    s = ExplanationStore(tmp_path / "e.jsonl")
    for _ in range(3):
        s.append(INPUTS, RESULT, VERS)
    lines = (tmp_path / "e.jsonl").read_text().splitlines()
    (tmp_path / "e.jsonl").write_text(lines[0] + "\n" + lines[2] + "\n")
    v = s.verify()
    assert v["ok"] is False and v["first_bad_index"] == 1 and "chain" in v["reason"]


def test_purge_drops_only_old_records_and_keeps_the_chain_valid(tmp_path):
    s = ExplanationStore(tmp_path / "e.jsonl")
    for _ in range(4):
        s.append(INPUTS, RESULT, VERS)
    # age the first two records by rewriting their timestamps AND re-hashing, as retention would see genuinely old records
    from src.modeling.explanation_store import _digest
    recs = [json.loads(x) for x in (tmp_path / "e.jsonl").read_text().splitlines()]
    old = (datetime.now(UTC) - timedelta(days=120)).isoformat(timespec="seconds")
    prev = GENESIS
    for i, r in enumerate(recs):
        if i < 2:
            r["ts"] = old
        r["prev"] = prev
        r["hash"] = _digest(r)
        prev = r["hash"]
    (tmp_path / "e.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert s.verify()["ok"]
    assert s.purge_older_than(90) == 2
    assert s.verify()["ok"] and s.verify()["records"] == 2


def test_store_never_keeps_fields_outside_the_intake_schema(tmp_path):
    s = ExplanationStore(tmp_path / "e.jsonl")
    rid = s.append(INPUTS, RESULT, VERS)
    banned = {"name", "account", "narrative", "tags", "tag_older", "tag_servicemember", "ssn", "email", "phone", "zip"}
    assert not banned & set(s.get(rid)["inputs"])


@pytest.fixture()
def store_client(tiny, tmp_path, monkeypatch):  # noqa: F811
    import joblib
    from fastapi.testclient import TestClient
    d, pipe = tiny
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic").fit(d[C.RAW_INPUT], d["y"])
    for t in C.TARGETS:
        joblib.dump({"calibrated": cal, "base": pipe, "target": t, "features": C.RAW_INPUT, "algorithm": "lightgbm",
                     "review_threshold": 0.3, "trained_at": "2026-01-01T00:00:00+00:00"}, tmp_path / f"risk_{t}.joblib")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("EXPLANATION_STORE_PATH", str(tmp_path / "store" / "explanations.jsonl"))
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_every_prediction_is_stored_and_retrievable_by_id(store_client):
    r = store_client.post("/predict", json=GOOD)
    rid = r.json()["explanation_id"]
    assert rid
    rec = store_client.get(f"/explanations/{rid}").json()
    assert rec["relief_probability"] == r.json()["relief_probability"] and rec["inputs"]["company"] == "acme bank"
    assert rec["model_versions"] == r.json()["model_versions"]
    assert store_client.get("/explanations/verify").json()["ok"] is True
    assert store_client.get("/explanations/doesnotexist").status_code == 404


def test_store_disabled_by_default_returns_no_id_and_503(tiny, tmp_path, monkeypatch):  # noqa: F811
    import joblib
    from fastapi.testclient import TestClient
    d, pipe = tiny
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic").fit(d[C.RAW_INPUT], d["y"])
    for t in C.TARGETS:
        joblib.dump({"calibrated": cal, "base": pipe, "target": t, "features": C.RAW_INPUT, "algorithm": "lightgbm",
                     "review_threshold": 0.3, "trained_at": "2026-01-01T00:00:00+00:00"}, tmp_path / f"risk_{t}.joblib")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("EXPLANATION_STORE_PATH", raising=False)
    from api.main import app
    with TestClient(app) as c:
        assert c.post("/predict", json=GOOD).json()["explanation_id"] is None
        assert c.get("/explanations/verify").status_code == 503


def test_metrics_endpoint_counts_predictions_without_leaking_inputs(store_client):
    store_client.post("/predict", json=GOOD)
    store_client.post("/predict", json={**GOOD, "company": "never seen co"})
    text = store_client.get("/metrics").text
    assert "risk_api_predictions_total" in text and "risk_api_predict_seconds_count" in text
    assert "risk_api_unseen_category_requests_total" in text
    assert "acme bank" not in text and "never seen co" not in text
