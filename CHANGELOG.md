# Changelog
All notable changes follow Semantic Versioning and Keep a Changelog.

## [1.0.0] - 2026-11-20
### Added
- API explanation store: every `/predict` is recorded in a hash-chained log (`src/modeling/explanation_store.py`); `GET /explanations/{id}`, `GET /explanations/verify`; Prometheus `GET /metrics`.
- Monitoring (`src/monitoring/`, `monitoring_app/`): two-stage job (model environment scores, pandas-2 environment runs Evidently and Fairlearn), alert rules traced to the Model Card, F2 gaps by tag with bootstrap intervals and by product, company-size tier and region, monitoring dashboard.
- Retraining (`src/modeling/retrain.py`): trigger rules, challenger build, twelve promotion gates, MLflow `challenger` alias, approver-gated promotion with archive. A dry run on historical windows produced a challenger that was more accurate but was blocked by the regional fairness gate.
- `docker-compose.yml` (API, stakeholder dashboard, monitoring dashboard, Prometheus, Grafana; pipeline and monitor-job on demand), `Dockerfile.monitoring`, `requirements-monitoring.txt`, Prometheus alert rules, Grafana dashboard.
- CI: monitoring job (pandas 2 and Evidently), dashboards job covers the monitoring app, compose smoke test.
- `docs/final/`: reconciliation log (15 cross-module corrections) and the governance pack.
### Changed
- Dashboard: record count is the published 535,578 (not the 535,601 pre-suppression extract); the G5 claim is removed (G5 descoped, G3 implemented but not back-tested); the six pre-registered fairness limits are recomputed at the deployed region lines; a SHAP-versus-LIME agreement table is added.
### Known issues
- Ethics panel sign-off is still pending; there is no panel for this coursework. F3 (proxy test) remains unmet. The relief model still does not beat the logistic baseline.

## [0.5.0] - 2026-11-02
### Added
- Module 5 stakeholder dashboard (`app/dashboard.py`, `app/views/`): Overview, What drives predictions, Try a complaint (what-if with side-by-side scenarios and one-field sweeps), Ethical Compliance (Fairlearn metrics from published aggregates, interactive review-line slider, bias found/addressed, proxy test) and Limits and transparency.
- `src/modeling/scoring.py`: one scoring implementation shared by the API and the dashboard.
- `src/dashboard/`: `build_assets.py` (aggregated, complaint-free data file `app/assets/dashboard_data.json`) and `fair.py` (Fairlearn MetricFrame on weighted confusion counts).
- `scripts/api_demo.py` and a Postman collection; `app/requirements.txt` for Streamlit Community Cloud; `.streamlit/config.toml`.
- 13 dashboard tests; CI `dashboard` job; Make targets.
### Changed
- `api/main.py` delegates scoring to `src.modeling.scoring.Scorer` (responses unchanged; 23 model tests still pass).
- `Dockerfile.dashboard` rebuilt for the new app.

## [0.4.0] - 2026-11-01
### Added
- Module 4 modelling package `src/modeling/` (time-aware split, leakage guard, baseline, logistic regression / random forest / LightGBM with cross-validated random search, isotonic calibration, MLflow tracking and registry).
- Explainability (SHAP, LIME, DiCE counterfactuals), Fairlearn audit with per-region equal-opportunity mitigation, sensitivity analysis, paired bootstrap against the logistic baseline.
- FastAPI service `api/main.py` with `/predict`, `/health`, `/model-info`; `Dockerfile.api`; `requirements-model.txt`, `requirements-api.txt`.
- Model Card, Fairness Metrics Report, pre-registered success criteria, executed SHAP notebook, MLflow screenshots and run export.
- 19 model and API tests; CI now has separate pipeline, model and docker jobs.
### Changed
- CI pipeline job excludes `tests/test_modeling.py` (different dependency set, tested in the model job).
### Known issues
- Relief model does not beat the logistic-regression baseline on the test window (PR-AUC difference -0.011, 95% CI -0.015 to -0.007); documented in the Model Card.

## [0.1.0] - 2026-11-01
### Added
- Repository scaffold from the Module 1 structure, environment files, CI workflow.
- CFPB ingestion with SHA-256 snapshot hash, data-contract checks, weekly baseline and surge flags (with tests).
- Data dictionary, RAID log, decision record ADR-0001, traceability matrix, sprint board definition.
