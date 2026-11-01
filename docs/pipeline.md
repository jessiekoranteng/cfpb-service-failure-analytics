# Data pipeline (Module 3)

Raw CFPB complaints go in; an anonymised, validated, model-ready table and a dashboard feed come out.
Orchestrated by Prefect (`flows/ccdb_pipeline.py`), validated by Great Expectations, checked for representation bias with Fairlearn.

```
ingest -> validate_raw -> clean -> transform -> integrate -> validate_curated -> bias_suite -> anonymise -> publish
                                                                 \-> dashboard_feed -----------------------/
```

| Stage | Module | What it does | Gate |
|---|---|---|---|
| ingest | `src/ingestion/ccdb_ingest.py` | Streams the 361 MB bulk ZIP in chunks, keeps the two in-scope products from 2020-01-01, 12 columns, records the source SHA-256 | retries x2 |
| validate_raw | `src/pipeline/validate.py` | 20 Great Expectations checks on the raw scope (schema, domains, nulls, date range, volume) | critical failure stops the run |
| clean | `src/pipeline/clean.py` | Trim, parse dates, drop bad/future dates, drop duplicates, fill Sub-product/Sub-issue/State with `Unknown`, leave Tags missing (missing = no tag recorded) | counts reported |
| transform | `src/pipeline/transform.py` | Targets `monetary_relief` and `untimely`, `week_start`, `year`, `company_key`, protected-tag flags (audit only) | |
| integrate | `src/pipeline/integrate.py` | Joins `reference/state_region.csv`; adds the company's 4-week-lagged complaint volume | |
| validate_curated | `src/pipeline/validate.py` | 16 checks on the curated frame (labels, region domain, lag range, tag flags) | critical failure stops the run |
| bias_suite | `src/pipeline/bias.py` | Coverage, representation drift, differential missingness, Fairlearn outcome disparity, company concentration | BLOCK stops the run; WARN is reported |
| anonymise | `src/pipeline/anonymise.py` | HMAC pseudonym key, PII scan and redaction, k-anonymity (k>=5), drops direct identifiers | any PII left stops the run |
| publish | `flows/ccdb_pipeline.py` | Writes `analysis_table.parquet` (no protected fields), `audit_groups.parquet` (restricted), `dashboard_weekly_feed.parquet` | |

## Outputs

| File | Audience | Contains |
|---|---|---|
| `data/processed/analysis_table.parquet` | analysts, Module 4 model | pseudonymous key, model features, two labels. No company name, state, date, Complaint ID or protected tags |
| `data/processed/audit_groups.parquet` | ethics panel only | pseudonymous key, year, region, protected-tag flags, labels (for the Module 4 fairness audit) |
| `data/processed/dashboard_weekly_feed.parquet` | dashboard (Module 5) | weekly company-issue counts with surge flags |
| `reports/pipeline_run.json`, `bias_report.json`, `lineage.json` | reviewers | run summary, bias results, stage-by-stage lineage with hashes |
| `logs/privacy_audit.jsonl` | auditor | hash-chained log of every access and transformation (no data values) |
| `gx/uncommitted/data_docs/local_site/index.html` | reviewers | Great Expectations Data Docs |

## Run it

```bash
pip install -r requirements-pipeline.txt
export PIPELINE_PSEUDONYM_KEY=<secret>            # required when PIPELINE_ENV=prod
python -m flows.ccdb_pipeline --source data/raw/complaints.csv.zip
python -m flows.ccdb_pipeline --serve --cron "0 6 * * *"   # daily schedule
pytest -q
docker build -t cfpb-pipeline . && docker run --rm -e PIPELINE_PSEUDONYM_KEY=... -v "$PWD/data/raw:/app/data/raw:ro" -v "$PWD/out:/out" cfpb-pipeline
```

## Scalability and collaboration

* Every stage is a pure function over a DataFrame, so the same code can move to Dask or Spark, or a warehouse, without changing the DAG shape.
* Thresholds live in `src/pipeline/config.py`; roles and retention live in `governance/policy.yaml`. Changing either is a reviewed pull request.
* CI runs lint, tests and a Docker build on every pull request, so a collaborator cannot merge a pipeline that fails its own checks.
* Lineage JSON and the hash-chained audit log let a new team member reproduce and verify any run.
