# Traceability matrix

Links each Module 1 goal to its tickets, code and tests.

| Goal | Description | Tickets | Code | Tests |
|---|---|---|---|---|
| G1 | Predict costly outcomes (relief) | T07, T08, T10, T13, T15 | src/ingestion, quality, features, models | tests/test_ingest.py, tests/test_contracts.py |
| G2 | Predict missed deadlines | T14 | src/models | planned: out-of-time evaluation test |
| G3 | Detect service-failure surges | T09, T20 | src/baseline, app | tests/test_baseline.py |
| G4 | Fair prioritisation | T16, T18, T23, T24 | src/fairness, docs | planned: equalized-odds gap test |
| G5 | Better triage policy | T19, T21 | src/triage_sim | planned: simulation regression test |
