"""Paths and thresholds for the pipeline. Every path is derived from one root so tests can run in a temp directory."""
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Thresholds (documented in docs/pipeline.md and the governance policy)
K_ANONYMITY_MIN = 5            # smallest allowed size of a quasi-identifier group in the published table
MIN_GROUP_N_WARN = 1_000       # audit group smaller than this: statistics flagged as low-power
MIN_GROUP_N_BLOCK = 100        # audit group smaller than this: publication blocked
SHARE_DRIFT_MAX = 0.05         # max absolute change of a group's yearly share vs its overall share
MISSINGNESS_GAP_MAX = 0.10     # max spread of Tags-missing rate across a grouping
PARITY_RATIO_MIN = 0.80        # four-fifths screen on outcome rates (diagnostic, not enforced)
TOP5_COMPANY_SHARE_WARN = 0.40


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def audit_log(self) -> Path:
        return self.logs / "privacy_audit.jsonl"

    @property
    def reference(self) -> Path:
        return PROJECT_ROOT / "reference"

    @property
    def policy(self) -> Path:
        return PROJECT_ROOT / "governance" / "policy.yaml"

    def make(self) -> "Paths":
        for p in (self.raw, self.interim, self.processed, self.reports, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


def actor() -> str:
    """Who is running the pipeline (recorded in the audit log). Set PIPELINE_ACTOR in scheduled runs."""
    return os.environ.get("PIPELINE_ACTOR") or os.environ.get("USER") or "unknown"
