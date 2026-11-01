"""Stream the CFPB bulk file, apply the Module 1 scope filter, and write a hashed, dated snapshot.

Scope (Vision Document, section 1): products below, received on or after 2020-01-01, 12 of 15 columns.
ZIP code, company public response and date sent to company are never read (data minimisation).
"""
import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PRODUCTS = ("Checking or savings account", "Money transfer, virtual currency, or money service")
START_DATE = "2020-01-01"
USECOLS = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue", "Company", "State",
    "Tags", "Submitted via", "Company response to consumer", "Timely response?", "Complaint ID",
]


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def filter_chunk(chunk):
    """Apply the scope filter to one chunk (strings in, filtered frame out)."""
    chunk = chunk.copy()
    chunk["Date received"] = pd.to_datetime(chunk["Date received"])
    keep = chunk["Product"].isin(PRODUCTS) & (chunk["Date received"] >= START_DATE)
    return chunk[keep]


def ingest(zip_path, out_path, chunksize=500_000):
    zip_path, out_path = Path(zip_path), Path(out_path)
    parts, total = [], 0
    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as fh:
        for chunk in pd.read_csv(fh, usecols=USECOLS, chunksize=chunksize, dtype=str):
            total += len(chunk)
            parts.append(filter_chunk(chunk))
    df = pd.concat(parts, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    meta = {
        "source": "https://files.consumerfinance.gov/ccdb/complaints.csv.zip",
        "retrieved": datetime.now(timezone.utc).date().isoformat(),
        "source_sha256": sha256_file(zip_path),
        "source_rows": total,
        "scope_rows": len(df),
        "columns": USECOLS,
    }
    (out_path.parent / "snapshot.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(json.dumps(ingest(args.zip, args.out), indent=2))
