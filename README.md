# Digital Channel Service-Failure Impact Analytics

Predicts which banking complaints become costly (monetary relief) or late (untimely response), detects
service-failure surges, and tests a risk-ranked triage policy. Built for BAN6800 (Nexford University).

* **Data:** U.S. CFPB Consumer Complaint Database (public). Scope: *Checking or savings account* and
  *Money transfer, virtual currency, or money service*, received 2020-01-01 onward
  (531,004 closed complaints in the 1 Nov 2026 snapshot). No confidential employer data is used.
* **Plan:** see `docs/` (data dictionary, RAID log, decisions, traceability) and `docs/project_board/` (sprints and tickets).
* **Ethics:** see `ETHICS_CHARTER.md`. Protected-group tags are used only to audit, never as features.

## Quickstart
```bash
conda env create -f environment.yml && conda activate cfpb-sfa     # or: pip install -r requirements.txt
make test                                                          # pytest + ruff
python -m src.ingestion.ccdb_ingest --zip data/raw/complaints.csv.zip --out data/interim/scope.parquet
```
Download the bulk file from https://files.consumerfinance.gov/ccdb/complaints.csv.zip into `data/raw/` (DVC-tracked).

## Layout
`data/` (DVC) · `docs/` · `notebooks/` · `src/{ingestion,quality,baseline,features,models,fairness,triage_sim}/` ·
`app/` · `tests/` · `models/` · `.github/workflows/ci.yml` · `dvc.yaml` · `Dockerfile`
