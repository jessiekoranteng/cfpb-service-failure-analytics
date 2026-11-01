import pandas as pd
import pytest

from src.pipeline import anonymise as A


def test_pseudonym_is_deterministic_keyed_and_irreversible_looking():
    ids = pd.Series(["123", "124", "123"])
    a = A.pseudonymise(ids, key=b"k1")
    b = A.pseudonymise(ids, key=b"k2")
    assert a[0] == a[2] and a[0] != a[1]
    assert (a != b).all()
    assert a.str.len().eq(16).all() and not a.str.contains("123").any()


def test_prod_requires_key(monkeypatch):
    monkeypatch.delenv("PIPELINE_PSEUDONYM_KEY", raising=False)
    monkeypatch.setenv("PIPELINE_ENV", "prod")
    with pytest.raises(RuntimeError):
        A.pseudonym_key()


def test_scanner_finds_each_pii_type_and_redacts():
    df = pd.DataFrame({"t": ["mail me at jo@example.com", "ssn 123-45-6789", "call 233-555-1234",
                             "card 4111 1111 1111 1111", "zip 90210", "nothing here"]})
    found = A.scan_pii(df)
    for kind in ("email", "ssn", "phone", "card", "zip"):
        assert any(k.endswith(kind) for k in found), kind
    out, counts = A.redact_pii(df)
    assert A.scan_pii(out) == {} and counts
    assert out.loc[5, "t"] == "nothing here"


def test_self_test_recall_is_complete(curated):
    r = A.pii_self_test(curated)
    assert r["min_recall"] == 1.0
    assert r["false_positives_on_real_sample"] == 0


def test_k_anonymity_enforced():
    d = pd.DataFrame({"region": ["West"] * 20 + ["South"] * 2, "Product": "P", "Submitted via": "Web", "year": 2024})
    out, rep = A.enforce_k_anonymity(d, k=5)
    assert rep["k_before"] == 2 and rep["k_after"] >= 5 or len(out) == 20
    assert A.k_anonymity(out) >= 5


def test_anonymise_removes_direct_identifiers(curated):
    out, rep = A.anonymise(curated)
    for gone in ("Complaint ID", "State", "Date received", "Company", "Tags"):
        assert gone not in out.columns
    assert "complaint_key" in out.columns and out["complaint_key"].is_unique
    assert rep["pii_remaining"] == {}
    assert rep["k_anonymity"]["k_after"] >= 5


def test_numeric_and_date_columns_are_not_touched():
    import pandas as pd

    from src.pipeline.anonymise import redact_pii, scan_pii
    df = pd.DataFrame({"n": [12345, 98765], "d": pd.to_datetime(["2024-01-01", "2024-01-08"]), "t": ["ok", "a@b.com"]})
    out, counts = redact_pii(df, columns=["n", "d", "t"])
    assert out["n"].dtype == df["n"].dtype and out["d"].dtype == df["d"].dtype
    assert counts == {"t:email": 1}
    assert scan_pii(df, columns=["n", "d", "t"]) == {"t:email": 1}
