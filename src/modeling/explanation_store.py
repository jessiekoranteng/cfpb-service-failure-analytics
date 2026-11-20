"""Explanation store: an append-only, hash-chained record of every score the API issues and why.

An auditor, a complainant's reviewer or the ethics panel must be able to answer "what did the model say about this
complaint, which model version said it, and what were the reasons?" months later. Each `/predict` call therefore
writes one JSON line:

    id, ts, inputs (the nine intake fields), inputs_sha256, model_versions, the two probabilities, the flag and the
    region threshold used, the SHAP reason codes, unseen values, prev (hash of the previous record), hash

Records never contain names, account numbers, narratives or protected tags, because the API never receives them.
Records are chained like the Module 3 privacy audit log: editing or deleting a line breaks `verify`.

Retention: `purge_older_than(days)` rewrites the file without old records and keeps the chain consistent. Purging the
oldest records cannot be told apart from never having had them; that is the cost of retention and is documented in
docs/final/ethical_ai_framework.md.
"""
import hashlib
import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

GENESIS = "0" * 64
RETENTION_DAYS = 90


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _digest(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256(_canon(body).encode()).hexdigest()


class ExplanationStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _records(self):
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def _last_hash(self) -> str:
        recs = self._records()
        return recs[-1]["hash"] if recs else GENESIS

    def append(self, inputs: dict, result: dict, model_versions: dict) -> str:
        """Store one scored request and return its id."""
        with self._lock:
            rec = {
                "id": uuid4().hex,
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "inputs": inputs,
                "inputs_sha256": hashlib.sha256(_canon(inputs).encode()).hexdigest(),
                "model_versions": model_versions,
                "relief_probability": result["relief_probability"],
                "untimely_probability": result["untimely_probability"],
                "flag_for_review": result["flag_for_review"],
                "review_threshold": result["review_threshold"],
                "reasons": result["reasons"],
                "unseen_values": result["unseen_values"],
                "prev": self._last_hash(),
            }
            rec["hash"] = _digest(rec)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            return rec["id"]

    def get(self, rid: str) -> dict | None:
        return next((r for r in self._records() if r["id"] == rid), None)

    def verify(self) -> dict:
        """Check every record's own hash and its link to the previous record. The first record's `prev` is the anchor."""
        recs = self._records()
        for i, r in enumerate(recs):
            if r["hash"] != _digest(r):
                return {"ok": False, "records": len(recs), "first_bad_index": i, "reason": "record content does not match its hash"}
            if i and r["prev"] != recs[i - 1]["hash"]:
                return {"ok": False, "records": len(recs), "first_bad_index": i, "reason": "chain link broken (a record was removed or reordered)"}
        return {"ok": True, "records": len(recs), "first_bad_index": None, "reason": None}

    def purge_older_than(self, days: int = RETENTION_DAYS, now: datetime | None = None) -> int:
        """Drop records older than the retention period; the first kept record's `prev` becomes the new anchor."""
        cutoff = (now or datetime.now(UTC)) - timedelta(days=days)
        with self._lock:
            recs = self._records()
            keep = [r for r in recs if datetime.fromisoformat(r["ts"]) >= cutoff]
            with open(self.path, "w", encoding="utf-8") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in keep)
            return len(recs) - len(keep)

    def inputs_frame(self):
        """Live traffic as rows of intake fields plus scores, for the drift monitor."""
        import pandas as pd
        rows = [{**r["inputs"], "relief_probability": r["relief_probability"], "flag_for_review": r["flag_for_review"],
                 "ts": r["ts"]} for r in self._records()]
        return pd.DataFrame(rows)
