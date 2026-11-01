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
`app/` · `api/` · `tests/` · `models/` · `.github/workflows/ci.yml` · `dvc.yaml` · `Dockerfile` · `Dockerfile.api`

## Module 3: data pipeline

Prefect DAG (`flows/ccdb_pipeline.py`), Great Expectations suites, Fairlearn bias suite, anonymiser, governance policy, hash-chained privacy audit log, 46 pytest tests and a Dockerfile.
See [docs/pipeline.md](docs/pipeline.md), [docs/anonymisation_plan.md](docs/anonymisation_plan.md), [docs/governance_framework.md](docs/governance_framework.md).
Evidence from the full run on 535,601 records is in [docs/evidence/](docs/evidence/) and [docs/figures/](docs/figures/).

```bash
pip install -r requirements-pipeline.txt
export PIPELINE_PSEUDONYM_KEY=<secret>
python -m flows.ccdb_pipeline --source data/raw/complaints.csv.zip
pytest -q
```

## Module 4: predictive model, explainability and fairness

Two LightGBM risk models (monetary relief; late response) trained on a time-aware split of the Module 3 table, tracked in MLflow
(51 runs, registry alias `champion`), explained with SHAP, LIME and DiCE, audited with Fairlearn, and served by a FastAPI `/predict` endpoint.
**Start with [MODEL_CARD.md](MODEL_CARD.md)**, then [docs/module4/fairness_metrics_report.md](docs/module4/fairness_metrics_report.md),
[docs/module4/success_criteria.md](docs/module4/success_criteria.md) (written before training) and
[notebooks/shap_analysis.ipynb](notebooks/shap_analysis.ipynb) (HTML copy alongside).

| Item | Where |
|---|---|
| Serialized models | `models/risk_monetary_relief.joblib`, `models/risk_untimely.joblib` |
| API | `api/main.py` (`GET /health`, `GET /model-info`, `POST /predict`); `Dockerfile.api` |
| Experiment tracking | `reports/m4/mlflow_runs_export.csv` (all runs, params, metrics, timestamps); screenshots in `docs/figures/module4/` |
| Figures and metrics | `reports/m4/` (ROC/PR/confusion curves, calibration, SHAP, LIME, fairness, sensitivity, JSON metrics, `facts.json`) |
| Tests | `tests/test_modeling.py` (19 tests) |

```bash
pip install -r requirements-model.txt
python -m src.modeling.train                 # tune, track in MLflow, calibrate, register, serialize   (about 15-20 minutes on 2 CPUs)
python -m src.modeling.analysis monetary_relief && python -m src.modeling.analysis untimely   # test window, fairness, mitigation, sensitivity
python -m src.modeling.adopt_mitigation      # store per-region thresholds in the relief bundle
python -m src.modeling.compare               # paired bootstrap vs logistic regression; exact top-20% recall
python -m src.modeling.facts && python -m src.modeling.docs_gen    # regenerate MODEL_CARD.md and the fairness report from the saved results
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5055
uvicorn api.main:app --port 8000             # http://127.0.0.1:8000/docs
make model-test
```

Example request (the response carries the calibrated probabilities, the review flag, the top-3 SHAP reasons and any category the model has not seen):

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{"product":"Checking or savings account",
  "sub_product":"Checking account","issue":"Managing an account","sub_issue":"Fee problem","channel":"Web","region":"South",
  "company":"BANK OF AMERICA NATIONAL ASSOCIATION","company_volume_lag4":900,"month":5}'
```

Do not run `train --quick` in this working tree: it overwrites `models/` and `reports/m4/`. Use `--uri` and a scratch copy.
