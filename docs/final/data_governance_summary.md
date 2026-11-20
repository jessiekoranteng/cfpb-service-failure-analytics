# Data Governance and Privacy Summary

One-page summary tying the Module 3 governance work to the Final Project components. Full documents: `docs/governance_framework.md`, `docs/anonymisation_plan.md`, `governance/policy.yaml`, `docs/pipeline.md`.

## 1. Data inventory

| Asset | Content | Layer | Where | Retention |
|---|---|---|---|---|
| CFPB bulk file | 18,849,258 public complaint records, 12 loaded columns | Raw | `data/raw` (not in Git) | Course plus 12 months |
| In-scope table | 535,601 records (two products, 2020 on) | Curated | Pipeline stage output | 90 days interim |
| `analysis_table.parquet` | 535,578 rows, 13 columns, HMAC pseudonym key, region not state, week not date, no protected fields | Published | `data/processed` | Policy-driven |
| `audit_groups.parquet` | Protected tag flags | Restricted | `data/processed`, audit role only | Policy-driven |
| Model bundles | Calibrated LightGBM, thresholds | Model | `models/`, registry | Until retired |
| Explanation Store | 9 intake fields, scores, reasons, model version, hash chain | Operational | API volume | 90 days |
| Monitoring files | Aggregates only | Reporting | `monitoring_app/assets` | Until superseded |
| Dashboard data | Aggregates, four worked examples built from public records | Reporting | `app/assets` | Until superseded |

Reconciliation C1: 535,601 is the in-scope extract before suppression; 535,578 is what the pipeline published (23 rows suppressed to keep k-anonymity at 5).

## 2. Controls

* **Minimisation:** 12 of 15 columns; no ZIP, narrative, company public response or send date.
* **Pseudonymisation:** complaint id to HMAC-SHA256 (key from environment); state to region; date to week.
* **Anonymisation checks:** regex scan for email, SSN, card, phone and ZIP patterns with a Faker recall test of at least 95%; k-anonymity k >= 5.
* **Access:** five roles and a least-privilege matrix; denial raises `PermissionError` and is logged.
* **Audit:** hash-chained privacy audit log (17 events on the full run, chain verified) and the new explanation-store chain.
* **Quality gates:** Great Expectations raw suite (20 checks) and curated suite (17 checks); critical failure stops publication.
* **Restricted tags:** Older American and Servicemember flags live only in the audit table and never reach the model; a unit test asserts the forbidden-column list.

## 3. Bias-detection results from the pipeline (Module 3, result WARN, nothing blocked)

| Check | Finding |
|---|---|
| Coverage | Smallest protected group 8,722 records; Email channel only 7 records (WARN) |
| Drift | Region and channel mix shifts year to year, up to 22 points on Web |
| Missingness | Tag blank rate varies from 3% to 57% by channel; tags recorded on 15.1% of complaints |
| Outcomes | Four-fifths screen flags region and tag gaps in raw relief rates (diagnostic only) |
| Concentration | Top 5 of 1,364 companies hold 45.8% of complaints |

These findings are carried into the Final Project as risks R9 (audit covers tagged complaints only) and R10 (skill concentrated in company identity), neither of which was in the Module 1 register (reconciliation C14). The model-level audit in Module 4 and the monthly monitoring job build on them.

## 4. Privacy position of the new components

| Component | Personal data? | Notes |
|---|---|---|
| API `/predict` | The nine intake fields only; no identifiers are accepted | Schema test asserts no name, account, narrative, tag, email, phone, ZIP key can be stored |
| Explanation Store | As above, with a request id | 90-day purge; chain re-anchors after purge |
| `/metrics` | None | Counters only; test asserts input values are never exposed |
| Monitoring dashboard | None | Aggregates; cells under 10 suppressed |
| Evidently report | Aggregated distributions, including company names (public) | |

## 5. Known gaps

The public demo has no authentication or access log; residency and role-based access promised in Module 1 are documented deviations (C9); retention for real bank data is not defined; no data-protection impact assessment has been done.
