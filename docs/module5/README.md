# Module 5: stakeholder dashboard, deck and API demo

| Deliverable | Where |
|---|---|
| Dashboard (Streamlit) | `app/dashboard.py`, `app/views/`, `app/common.py`; requirements `app/requirements.txt`; theme `.streamlit/config.toml` |
| Aggregated data the dashboard reads | `app/assets/dashboard_data.json` (rebuild: `python -m src.dashboard.build_assets`) |
| Fairlearn on aggregates | `src/dashboard/fair.py` (MetricFrame with sample weights over confusion counts) |
| Shared scorer (API and dashboard) | `src/modeling/scoring.py` |
| API demo | `scripts/api_demo.py`; output `reports/m5/api_demo_output.json`; Postman: `postman/cfpb_risk_api.postman_collection.json` |
| Tests | `tests/test_dashboard.py` (14 tests) |

## Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with the GitHub account that owns the repository.
2. **Create app** -> **Deploy a public app from GitHub**.
3. Repository `jessiekoranteng/cfpb-service-failure-analytics`, branch `main`, main file path `app/dashboard.py`.
4. **Advanced settings**: Python version **3.11**. (Streamlit reads `app/requirements.txt` because it sits next to the entrypoint.)
5. Choose an app URL (for example `cfpb-service-failure`), then **Deploy**. The first build takes several minutes (shap and lightgbm).
6. Open the URL, click through all five pages, and put it on the deck's Live Dashboard URL slide.

## Design decisions

* **No complaint-level data online.** The dashboard needs only group counts, 0.0025-wide score histograms per group and a drop-down vocabulary.
* **One scorer.** The what-if page and `POST /predict` call the same `Scorer`; a test asserts identical numbers.
* **Review-line slider is exact.** Thresholds move in steps of 0.0025, the width of the stored histogram bins; a test checks the histogram reproduces the exact confusion counts at the global line.
* **Misses stay visible.** The Ethical Compliance page and the goal scorecard show unmet targets (Charter F2 four-group, F3, logistic baseline, G1 recall target).

## Failure modes

* Retraining without rebuilding `dashboard_data.json` leaves the fairness charts describing the old model. `test_example_scores_match_the_scorer` fails in CI in that case.
* Community Cloud puts idle apps to sleep; the first visit after a sleep takes about a minute.
* Streamlit Cloud has about 1 GB of memory; SHAP is imported lazily and only for the three reasons on one complaint.
