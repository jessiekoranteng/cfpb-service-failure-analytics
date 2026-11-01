# Data Dictionary

Source: U.S. CFPB Consumer Complaint Database, bulk snapshot of 1 November 2026, scoped to *Checking or savings account* and *Money transfer, virtual currency, or money service*, received on or after 2020-01-01. Records in scope: 531,004 closed (plus in-progress). Field definitions: https://cfpb.github.io/api/ccdb/fields.html

## 1. Source columns loaded (12 of 15)

| Column | Type | Description | Nulls % | Distinct | Role in project | Notes |
|---|---|---|---|---|---|---|
| Date received | date | Date the CFPB received the complaint. | 0.0 | 2,495 | Time axis; baseline; week-of-year and lag features | No nulls. Range 2020-01-01 to 2026-10-31 in the 1 Nov 2026 snapshot. |
| Product | categorical | Type of product the consumer identified. | 0.0 | 2 | Stratifier; feature | Scope filter keeps 2 products. |
| Sub-product | categorical | Type of sub-product the consumer identified. | 0.01 | 16 | Feature | Not all products have sub-products; nulls become 'Unknown'. |
| Issue | categorical | Issue the consumer identified; values depend on Product. | 0.0 | 31 | Feature; surge-monitor key | Strongest single signal: relief rate ranges 2.6% to 16.8% across issues with >= 2,000 complaints. |
| Sub-issue | categorical | Sub-issue the consumer identified; depends on Product and Issue. | 34.2 | 51 | Feature | Nulls become 'Unknown'. |
| Company | categorical | Company the complaint is about. | 0.0 | 1,368 | Grouping key (baseline); feature tested with and without (risk R-01) | High cardinality; top 5 firms hold 45.9% of volume. |
| State | categorical | State of the consumer's mailing address (includes territories and military codes). | 2.55 | 62 | Mapped to region; raw state never used as a feature | Nulls become 'Unknown'. |
| Tags | categorical | Labels such as 'Older American' (age 62+) and 'Servicemember'. | 84.9 | 3 | AUDIT ONLY (fairness objective F1); never a feature | Missing on 84.9% of records; absence means untagged, not 'not older'. |
| Submitted via | categorical | How the complaint was submitted to the CFPB. | 0.0 | 6 | Feature | Web 86.0% of records. |
| Company response to consumer | categorical | How the company responded, e.g. 'Closed with monetary relief'. | 0.0 | 5 | LABEL SOURCE for monetary_relief; never a feature (leakage) | 'In progress' rows are excluded from model training. |
| Timely response? | categorical | Whether the company gave a timely response (Yes/No). | 0.0 | 2 | LABEL SOURCE for untimely; never a feature (leakage) | Positive class is rare (about 1.2%). |
| Complaint ID | integer (text key) | Unique identification number for a complaint. | 0.0 | 535,601 | Primary key | Unique in the snapshot; the only identifier retained. |

## 2. Derived fields

| Field | Type | Definition | Use |
|---|---|---|---|
| monetary_relief | binary | 1 if Company response to consumer = 'Closed with monetary relief', else 0; computed only for closed complaints. | Target for goal G1 |
| untimely | binary | 1 if Timely response? = 'No', else 0. | Target for goal G2 |
| week_start | date | Monday of the week of Date received. | Baseline and surge key (G3) |
| region | categorical | State mapped to a coarse region; territories and military codes to 'Other'. | Feature (replaces raw State) |
| company_volume_lag4 | numeric | Complaints for the company in the previous 4 weeks. | Feature; surge context |
| tag_older / tag_servicemember | binary | Flags parsed from Tags. | Fairness audit only (F1, F2) |

## 3. Source columns deliberately excluded

| Column | Reason |
|---|---|
| ZIP code | Quasi-identifier; re-identification risk (R-03). Never read into memory. |
| Company public response | Optional preset text added after the outcome; leakage risk and no planned use. |
| Date sent to company | Process timestamp with no planned use; excluded under data minimisation. |
| Consumer complaint narrative | Not present in the bulk file; free text would add re-identification risk. |

## 4. Label definitions and known caveats

* Closed complaints: 531,004. Monetary relief rate 11.1%; untimely rate 1.16%.
* Outcomes are company decisions recorded by the regulator, not ground-truth harm; they vary by company (risk R-01).
* Relief rate by year: 2020 14.6%, 2021 14.6%, 2022 14.1%, 2023 12.9%, 2024 13.4%, 2025 6.5%, 2026 11.9% (drift, risk R-07).
* The database is not a statistical sample and is not representative of all consumers (CFPB, 2026).
