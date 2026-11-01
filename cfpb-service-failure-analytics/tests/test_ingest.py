import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from src.ingestion.ccdb_ingest import PRODUCTS, USECOLS, filter_chunk, ingest

SAMPLE = Path(__file__).parent / "fixtures" / "ccdb_sample.csv"


def _bulk_zip(tmp_path):
    rows = pd.read_csv(SAMPLE, dtype=str)
    extra = rows.head(3).copy()
    extra["Product"] = "Mortgage"                       # out of scope
    old = rows.head(2).copy()
    old["Date received"] = "2019-06-01"                 # before scope start
    old["Complaint ID"] = ["900001", "900002"]
    extra["Complaint ID"] = ["900003", "900004", "900005"]
    full = pd.concat([rows, extra, old], ignore_index=True)
    full["ZIP code"] = "12345"                          # present in source, must never be loaded
    buf = io.StringIO()
    full.to_csv(buf, index=False)
    zp = tmp_path / "complaints.csv.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("complaints.csv", buf.getvalue())
    return zp, len(rows)


def test_filter_chunk_applies_scope():
    df = pd.read_csv(SAMPLE, dtype=str)
    assert set(filter_chunk(df)["Product"]) <= set(PRODUCTS)


def test_ingest_scope_columns_and_hash(tmp_path):
    zp, n_scope = _bulk_zip(tmp_path)
    out = tmp_path / "out" / "scope.parquet"
    meta = ingest(zp, out, chunksize=100)
    df = pd.read_parquet(out)
    assert len(df) == n_scope == meta["scope_rows"]
    assert list(df.columns) == USECOLS
    assert "ZIP code" not in df.columns
    assert len(meta["source_sha256"]) == 64
    assert json.loads((out.parent / "snapshot.json").read_text())["scope_rows"] == n_scope
