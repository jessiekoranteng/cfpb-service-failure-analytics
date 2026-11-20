# Datasheet: scoped CFPB Consumer Complaint extract

Format follows Gebru et al. (2021), *Datasheets for datasets* (https://doi.org/10.1145/3458723). Written for the Final Project; Module 1 promised it (ticket T18) and it was not produced until now. Every figure below is taken from `docs/data_dictionary.md`, `docs/evidence/bias_report.json` and the Module 3 pipeline run.

## Motivation
* **Purpose.** Demonstrate an explainable complaint-prioritisation method for a retail bank's service quality committee. The data were not created for this purpose.
* **Creator and funder.** U.S. Consumer Financial Protection Bureau (CFPB), a public body; published as open data. This project's author scoped and processed a copy; no funder.
* **Gap filled.** A real, public, outcome-labelled complaint file stands in for a bank's own records, which could not be used without written approval and Data Protection Act 2012 compliance.

## Composition
* **Instances.** One consumer complaint each. Bulk file: 18,849,258 records. Scope: Checking or savings account and Money transfer, virtual currency or money service, received 2020-01-01 or later: 535,601 records; 535,578 after 23 were suppressed to hold k-anonymity at 5.
* **Fields.** 12 of 15 source columns loaded (see data dictionary). Excluded: ZIP code, company public response, date sent to company. The complaint narrative is not in the bulk file.
* **Labels.** `monetary_relief` (company closed with monetary relief) and `untimely` (company response not timely), both derived from company-reported fields. They are a company's decision and self-report, not ground truth.
* **Missing data.** Sub-issue 34.2% null (set to Unknown); State 2.55%; Tags 84.9% null, so protected-group audits cover about 15% of complaints, with the blank rate varying from 3% to 57% by channel.
* **Sensitive content.** Tags can mark Older American (age 62 or over) and Servicemember. Used only for auditing; held in a restricted table; never a model input.
* **Imbalance and concentration.** Relief about 11% of closed complaints; untimely about 1.2%. Five companies hold 45.8% of complaints; one company was 46% of January 2025.
* **Representativeness.** Only people who complained to a U.S. regulator, a self-selected group, and U.S. consumers. Not representative of Ghanaian bank customers.

## Collection
* **How.** Consumers submit to the CFPB, which routes the complaint to the company; the company's response is recorded. Narratives are published only with consent and are not in the bulk file.
* **When.** Records to 2026-10-31 in the snapshot of 1 November 2026, stored with its SHA-256 hash and never overwritten.
* **Consent and review.** Governed by CFPB publication policy. No ethical review by this project; the data are used as published.

## Preprocessing
Raw file validated by 20 Great Expectations checks; cleaning, label derivation, week and four-week company-volume lag; complaint id replaced by an HMAC-SHA256 pseudonym; state mapped to region; date to week; PII scan; k-anonymity check; curated table validated by 17 checks. Raw data are kept read-only. Code: `flows/`, `src/pipeline/`, `src/quality/`.

## Uses
* **Intended.** Method demonstration and decision support for ordering human review of complaints.
* **Not suitable.** Deciding any individual complaint, redress or deadline; ranking staff or companies; inferring causes; claiming accuracy for Ghanaian or other non-U.S. populations; training on narrative or personal data.
* **Known risks.** Label bias (company decisions), proxy paths through company and region, partial tag coverage, volume surges that move the base rate (January 2025).

## Distribution
Source data: public, on the CFPB website. This repository does not redistribute the raw file (git-ignored, tracked by hash). Published aggregates and the pseudonymised table are for coursework; reuse should respect the CFPB's terms.

## Maintenance
Owner: project author. Refresh by re-running the pipeline on a new dated snapshot; each refresh is a new snapshot, never an overwrite. Schema changes are caught by the raw-file checks. Retention: raw snapshots for the course plus 12 months; interim data 90 days. Errata go in `CHANGELOG.md`.
