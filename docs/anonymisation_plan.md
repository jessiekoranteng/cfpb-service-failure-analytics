# Data anonymisation plan

**Source:** the public CFPB Consumer Complaint Database. Narratives are not ingested (the pipeline reads 12 of 15 columns), so free-text PII is out of scope by design; the scanner is a second line of defence.

| Field | Treatment | Reason |
|---|---|---|
| Complaint ID | Replaced by `complaint_key` = HMAC-SHA256(secret, ID), truncated; secret from `PIPELINE_PSEUDONYM_KEY` | Unlinkable without the key, stable across runs for joins |
| Company | Dropped as raw text; the table keeps `company_key`, a slug of the public company name, because firm-level volume is a model feature and the dashboard goal. Companies are organisations, not individuals | Direct identifier of a person is never present; firm names are public in the source |
| State | Replaced by 5-way region | Quasi-identifier generalisation |
| Date received | Replaced by `week_start` and `year` | Quasi-identifier generalisation |
| Tags (Older American, Servicemember) | Removed from model table; kept as flags only in the restricted audit table | Protected groups: fairness audit only, never a feature |
| Remaining text fields | Text columns only (numbers and dates are skipped so they keep their type) are scanned for email, SSN, card, phone and ZIP patterns and redacted | Defence in depth |

**k-anonymity:** after generalisation every combination of quasi-identifiers must occur at least k=5 times; rarer groups are suppressed and counted in the run report.

**Verification:** `pii_self_test` generates synthetic PII with Faker and requires the scanner to find at least 95% of it; unit tests cover each pattern, key handling, determinism and suppression. The run fails if any PII pattern remains.

**Key management:** in production the key is injected as an environment secret and never committed; development uses a clearly labelled dev key and refuses to run as `prod` without a real one.
