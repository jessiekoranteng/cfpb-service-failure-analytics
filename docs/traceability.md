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
| G3, G5 | Not re-assessed / not assessed; stated as such | app/views/overview.py | tests/test_dashboard.py |
