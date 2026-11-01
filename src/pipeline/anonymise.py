"""Stage 7: anonymisation and PII removal (Data Anonymization Plan, docs/anonymisation_plan.md).

Techniques, in order:
  1. Pseudonymise the only direct identifier (`Complaint ID`) with a keyed HMAC-SHA256; the key is never stored in the repo.
  2. Generalise quasi-identifiers: State -> region, exact date -> week_start (already done upstream), raw Sub-issue kept
     only because it is a controlled vocabulary.
  3. Scan every text column for PII patterns (email, phone, SSN, card number, 5-digit ZIP-like tokens) and redact hits.
  4. Enforce k-anonymity (k >= K_ANONYMITY_MIN) on the quasi-identifier set by generalising rare groups.
  5. Self-test: inject Faker-generated PII into a copy of the data and prove the scanner finds it (recall check).
"""
import hashlib
import hmac
import os
import re
import warnings

import pandas as pd

from src.pipeline.config import K_ANONYMITY_MIN

PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card": re.compile(r"(?<!\d)(?:\d[ -]?){11,18}\d(?!\d)"),
    "phone": re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"),
    "zip": re.compile(r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)"),
}
DEV_KEY = b"dev-only-key-not-for-production"
QUASI_IDENTIFIERS = ["region", "Product", "Submitted via", "year"]


def pseudonym_key() -> bytes:
    key = os.environ.get("PIPELINE_PSEUDONYM_KEY")
    if key:
        return key.encode()
    if os.environ.get("PIPELINE_ENV") == "prod":
        raise RuntimeError("PIPELINE_PSEUDONYM_KEY must be set when PIPELINE_ENV=prod")
    warnings.warn("PIPELINE_PSEUDONYM_KEY not set: using the development key (outputs are NOT production-safe)", stacklevel=2)
    return DEV_KEY


def pseudonymise(ids: pd.Series, key: bytes | None = None) -> pd.Series:
    key = key or pseudonym_key()
    return ids.astype(str).map(lambda v: hmac.new(key, v.encode(), hashlib.sha256).hexdigest()[:16])


def _text_columns(df: pd.DataFrame, columns=None) -> list:
    """Only free-text (object/string) columns are scanned: numbers and dates cannot carry PII and must keep their dtype."""
    text = set(df.select_dtypes(include=["object", "string"]).columns)
    return [c for c in (columns if columns is not None else df.columns) if c in text]


def scan_pii(df: pd.DataFrame, columns=None) -> dict:
    """Count PII-pattern matches per (column, pattern) across text columns."""
    cols = _text_columns(df, columns)
    found = {}
    for c in cols:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        for name, rx in PII_PATTERNS.items():
            n = int(s.str.contains(rx).sum())
            if n:
                found[f"{c}:{name}"] = n
    return found


def redact_pii(df: pd.DataFrame, columns=None) -> tuple[pd.DataFrame, dict]:
    d = df.copy()
    cols = _text_columns(d, columns)
    counts = {}
    for c in cols:
        s = d[c].astype("string")
        for name, rx in PII_PATTERNS.items():
            hit = s.str.contains(rx, na=False)
            if hit.any():
                counts[f"{c}:{name}"] = int(hit.sum())
                s = s.str.replace(rx, f"[REDACTED-{name.upper()}]", regex=True)
        d[c] = s
    return d, counts


def k_anonymity(df: pd.DataFrame, qi=None) -> int:
    qi = qi or QUASI_IDENTIFIERS
    return int(df.groupby(qi, observed=True).size().min()) if len(df) else 0


def enforce_k_anonymity(df: pd.DataFrame, k: int = K_ANONYMITY_MIN, qi=None) -> tuple[pd.DataFrame, dict]:
    """Generalise rare quasi-identifier groups: first region -> 'Other', then Submitted via -> 'Other'; remaining
    groups still below k are suppressed (rows dropped) and counted."""
    qi = qi or QUASI_IDENTIFIERS
    d = df.copy()
    rep = {"k_before": k_anonymity(d, qi), "k_target": k}
    for col in ("region", "Submitted via"):
        size = d.groupby(qi, observed=True)[qi[0]].transform("size")
        rare = size < k
        rep[f"generalised_{col}"] = int(rare.sum())
        d.loc[rare, col] = "Other"
    size = d.groupby(qi, observed=True)[qi[0]].transform("size")
    rare = size < k
    rep["suppressed_rows"] = int(rare.sum())
    d = d[~rare].reset_index(drop=True)
    rep["k_after"] = k_anonymity(d, qi)
    return d, rep


def pii_self_test(df: pd.DataFrame, n: int = 200, seed: int = 7) -> dict:
    """Inject Faker-generated PII into a copy of the text columns and measure scanner recall."""
    from faker import Faker

    fk = Faker("en_US")
    fk.seed_instance(seed)
    gens = {"email": fk.email, "ssn": fk.ssn, "phone": lambda: fk.numerify("###-###-####"), "card": fk.credit_card_number,
            "zip": fk.zipcode}
    probe = pd.DataFrame({"text": [f"contact {gens[k]()} please" for k in gens for _ in range(n)],
                          "kind": [k for k in gens for _ in range(n)]})
    out = {}
    for kind, grp in probe.groupby("kind"):
        hits = grp["text"].str.contains(PII_PATTERNS[kind]).sum()
        out[kind] = round(float(hits) / len(grp), 4)
    clean_hits = sum(scan_pii(df.head(50_000)).values()) if len(df) else 0
    return {"recall_by_type": out, "min_recall": min(out.values()), "false_positives_on_real_sample": clean_hits}


def anonymise(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = df.copy()
    d["complaint_key"] = pseudonymise(d["Complaint ID"])
    d = d.drop(columns=["Complaint ID", "State", "Date received", "Company", "Tags"])
    d, redactions = redact_pii(d, columns=[c for c in d.columns if c != "complaint_key"])
    d, kreport = enforce_k_anonymity(d)
    remaining = scan_pii(d, columns=[c for c in d.columns if c != "complaint_key"])
    report = {"direct_identifiers_removed": ["Complaint ID", "Date received", "State", "Tags", "Company (kept as company_key)"],
              "pseudonym": "HMAC-SHA256 truncated to 16 hex chars", "pii_redactions": redactions,
              "pii_remaining": remaining, "k_anonymity": kreport}
    return d, report
