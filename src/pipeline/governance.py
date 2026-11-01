"""Data Governance Framework as code: role-based access control and retention enforcement.

The policy lives in governance/policy.yaml. Every decision (allowed or denied) is written to the privacy audit log.
"""
import time
from pathlib import Path

import yaml

from src.pipeline.audit import AuditLog


def load_policy(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def check_access(policy: dict, role: str, layer: str, action: str, log: AuditLog | None = None) -> bool:
    """Return True when `role` may perform `action` on `layer`; raise PermissionError otherwise. Both outcomes are logged."""
    if role not in policy["roles"]:
        allowed = False
    else:
        allowed = action in policy.get("access", {}).get(role, {}).get(layer, [])
    if log is not None:
        log.event("access_check", layer, f"role={role} action={action}", outcome="allowed" if allowed else "denied")
    if not allowed:
        raise PermissionError(f"role '{role}' may not {action} layer '{layer}'")
    return True


def enforce_retention(policy: dict, layer: str, root, now: float | None = None, dry_run: bool = True,
                      log: AuditLog | None = None) -> list[str]:
    """Delete (or, with dry_run, list) files in a layer older than its retention window."""
    now = time.time() if now is None else now
    days = policy["layers"][layer]["retention_days"]
    base = Path(root) / policy["layers"][layer]["path"]
    expired = []
    if base.exists():
        for f in sorted(p for p in base.rglob("*") if p.is_file() and p.name != ".gitkeep"):
            if (now - f.stat().st_mtime) / 86400 > days:
                expired.append(str(f.relative_to(root)))
                if not dry_run:
                    f.unlink()
    if log is not None:
        log.event("retention_enforced" if not dry_run else "retention_checked", layer, f"retention_days={days}",
                  rows_out=len(expired), detail={"files": expired, "dry_run": dry_run})
    return expired
