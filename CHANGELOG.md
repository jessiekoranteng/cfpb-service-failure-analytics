# Changelog
All notable changes follow Semantic Versioning and Keep a Changelog.

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
