# Data governance framework

Enforced in code by `src/pipeline/governance.py` from `governance/policy.yaml`.

## Access control (least privilege)

| Role | raw | interim | analysis table | audit table (restricted) | reports | audit log |
|---|---|---|---|---|---|---|
| data_engineer | read, write | read, write, delete | read, write, delete | read, write | read, write | read, append |
| analyst | - | - | read | - | read | - |
| ethics_panel | - | - | read | read | read | read |
| auditor | - | - | - | - | read | read |
| viewer | - | - | - | - | read | - |

`check_access(role, layer, action)` raises `PermissionError` on any request outside this matrix and writes the grant or denial to the privacy audit log.

## Retention

| Layer | Days | Why |
|---|---|---|
| raw | 365 | Reproduce any run for a year |
| interim | 90 | Rebuildable from raw |
| processed (analysis, audit) | 365 | Covers a model-retraining cycle |
| reports | 730 | Evidence for reviews |
| audit log | 730 | Longest evidence trail |

`enforce_retention(policy, root, dry_run=True)` lists what has expired; `dry_run=False` deletes it and logs each deletion.

## Accountability

* Owner reviews the policy every 90 days.
* Ethics panel gates: approves data before use, signs off the model, approves each dashboard release (see Module 2 architecture).
* Privacy audit log: JSON lines, no data values, each record carries the hash of the previous one; `verify_chain` detects edits, deletions or reordering.
