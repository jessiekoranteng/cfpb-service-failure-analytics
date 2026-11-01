"""Data lineage: every stage records its inputs, outputs, row counts, content hashes and the code that produced them."""
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def file_sha256(path, block=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash(*modules) -> str:
    h = hashlib.sha256()
    for m in modules:
        h.update(Path(m.__file__).read_bytes())
    return h.hexdigest()[:12]


class Lineage:
    def __init__(self, path):
        self.path = Path(path)
        self.steps: list[dict] = []

    def add(self, stage: str, inputs: list[str], outputs: list[str], rows_in=None, rows_out=None, output_sha256=None,
            code_version=None, detail=None):
        self.steps.append({
            "stage": stage, "inputs": inputs, "outputs": outputs, "rows_in": rows_in, "rows_out": rows_out,
            "output_sha256": output_sha256, "code_version": code_version,
            "ts": datetime.now(UTC).isoformat(timespec="seconds"), "detail": detail or {},
        })

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"steps": self.steps}, indent=2))
        return self.path
