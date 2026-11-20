# Traceability matrix

Links each Module 1 goal to its tickets, code and tests.

| Goal | Description | Tickets | Code | Tests |
|---|---|---|---|---|
| G1 | Predict costly outcomes (relief) | T07, T08, T10, T13, T15 | src/ingestion, quality, features, models, src/modeling, api | tests/test_ingest.py, tests/test_contracts.py, tests/test_modeling.py |
| G2 | Predict missed deadlines | T14 | src/models, src/modeling | tests/test_modeling.py (time-aware split, leakage guard) |
| G3 | Detect service-failure surges | T09, T20 | src/baseline, app | tests/test_baseline.py |
| G4 | Fair prioritisation | T16, T18, T23, T24 | src/fairness, src/modeling/fairness.py, docs/module4 | tests/test_modeling.py (planted-gap fairness test, region thresholds in API) |
| G5 | Better triage policy | T19, T21 | src/triage_sim | planned: simulation regression test |

## Module 5 additions
| Goal | Evidence in Module 5 | Code | Tests |
|---|---|---|---|
| G1 | Goal scorecard and what-if on the dashboard; deck slides 3-7 | app/views/overview.py, whatif.py, src/dashboard/build_assets.py | tests/test_dashboard.py |
| G2 | Goal scorecard (top-10% capture 66%, advisory) | src/dashboard/build_assets.py (goals) | tests/test_dashboard.py::test_goal_scorecard_covers_g1_to_g5_and_reports_misses |
| G4 | Ethical Compliance page; deck slides 8-9 | app/views/ethics.py, src/dashboard/fair.py | tests/test_dashboard.py (Fairlearn on aggregates reproduces Module 4) |
| G3, G5 | G3 implemented but not back-tested; G5 descoped (see reconciliation log C5) | app/views/overview.py | tests/test_dashboard.py |

## Final Project additions
| Requirement | Evidence | Code | Tests |
|---|---|---|---|
| Reconcile contradictions across modules | docs/final/reconciliation_log.md (C1 to C17) | src/dashboard/build_assets.py (deployed-line criteria, G3/G5), app/views/*.py | tests/test_dashboard.py (record count, deployed lines, G5 claim removed) |
| Live API with audit trail | `POST /predict` returns `explanation_id`; `GET /explanations/{id}`, `/explanations/verify`, `/metrics` | api/main.py, src/modeling/explanation_store.py | tests/test_explanation_store.py |
| Explanation Store | Hash-chained JSON lines, 90-day retention, tamper detection | src/modeling/explanation_store.py | tests/test_explanation_store.py (edit, delete, purge, schema) |
| Monitoring dashboard (Evidently, Fairlearn) | monitoring_app/ (Streamlit) and monitoring_app/assets/evidently_drift_report.html | src/monitoring/{export_scored,job,metrics,thresholds}.py, monitoring_app/dashboard.py | tests/test_monitoring.py, tests/test_monitoring_app.py |
| Live-traffic monitoring (Prometheus, Grafana) | monitoring/prometheus.yml, alerts.yml, grafana dashboard | docker-compose.yml | CI compose smoke test |
| Continuous fairness monitoring, F2 by product and company size | monitoring_app "Fairness" tab | src/monitoring/job.py (fairness_tags, fairness_cuts) | tests/test_monitoring.py (replay reproduces Module 4 F2 to 6 decimals) |
| Model retraining with MLflow | challenger alias, gate report, approver-gated promotion | src/modeling/retrain.py | tests/test_retrain.py |
| CI/CD | .github/workflows/ci.yml (pipeline, model, monitoring, dashboards, docker, compose smoke) | | |
| Docker Compose (pipeline + API + monitoring) | docker-compose.yml | Dockerfile, Dockerfile.api, Dockerfile.dashboard, Dockerfile.monitoring | `docker compose config`; CI smoke test |
| Governance pack | docs/final/*.md | | |
