"""Privacy audit logging for every data access and transformation.

Each event is one JSON line. Events never contain data values, only metadata: who, what, which dataset, how many
rows and which columns, and a content hash of the output. Events are hash-chained (each record stores the hash of the
previous one), so deleting or editing a line is detectable with `verify_chain`.
"""
import functools
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.pipeline.config import actor

logger = logging.getLogger("privacy_audit")
GENESIS = "0" * 64


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Order-independent content hash of a frame (used for lineage and the audit trail)."""
    h = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).values
    cols = ",".join(map(str, df.columns))
    return hashlib.sha256(h.tobytes() + cols.encode()).hexdigest()


class AuditLog:
    def __init__(self, path, who: str | None = None):
        self.path = Path(path)
        self.who = who or actor()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS
        with open(self.path, "rb") as fh:
            last = fh.readlines()[-1]
        return json.loads(last)["hash"]

    def event(self, action: str, dataset: str, purpose: str, rows_in=None, rows_out=None, columns=None,
              output_hash=None, outcome: str = "allowed", detail: dict | None = None) -> dict:
        rec = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "actor": self.who,
            "action": action,
            "dataset": dataset,
            "purpose": purpose,
            "rows_in": None if rows_in is None else int(rows_in),
            "rows_out": None if rows_out is None else int(rows_out),
            "columns": None if columns is None else list(map(str, columns)),
            "output_sha256": output_hash,
            "outcome": outcome,
            "detail": detail or {},
            "prev": self._last_hash(),
        }
        rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        logger.info("%s %s rows_in=%s rows_out=%s outcome=%s", action, dataset, rec["rows_in"], rec["rows_out"], outcome)
        return rec

    def audited(self, action: str, dataset: str, purpose: str):
        """Decorator for a function that takes a DataFrame first and returns a DataFrame (or (DataFrame, report))."""
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(df, *a, **kw):
                out = fn(df, *a, **kw)
                frame = out[0] if isinstance(out, tuple) else out
                self.event(action, dataset, purpose, rows_in=len(df), rows_out=len(frame), columns=frame.columns,
                           output_hash=frame_fingerprint(frame))
                return out
            return wrapper
        return deco


def verify_chain(path) -> tuple[bool, int]:
    """Return (ok, number_of_events). ok is False when any record was altered, removed or reordered."""
    prev, n = GENESIS, 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        claimed = rec.pop("hash")
        if rec["prev"] != prev or hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest() != claimed:
            return False, n
        prev, n = claimed, n + 1
    return True, n
