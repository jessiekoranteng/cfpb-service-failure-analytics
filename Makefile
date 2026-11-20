.PHONY: monitor retrain-dry-run monitoring-dashboard compose-config lint test pipeline docker model-lint model-test api docker-api dashboard dashboard-test docker-dashboard api-demo dashboard-data
lint:
	ruff check src flows tests
test: lint
	pytest -q --ignore=tests/test_modeling.py --ignore=tests/test_dashboard.py --ignore=tests/test_explanation_store.py --ignore=tests/test_retrain.py --ignore=tests/test_monitoring.py --ignore=tests/test_monitoring_app.py --ignore=tests/test_ablation.py
pipeline:
	PIPELINE_PSEUDONYM_KEY=$${PIPELINE_PSEUDONYM_KEY:-dev-key-not-for-prod} python -m flows.ccdb_pipeline --source data/raw/complaints.csv.zip
docker:
	docker build -t cfpb-pipeline .
model-lint:
	ruff check src/modeling src/monitoring api tests/test_modeling.py tests/test_explanation_store.py tests/test_retrain.py tests/test_monitoring.py tests/test_ablation.py
model-test: model-lint
	pytest -q tests/test_modeling.py tests/test_explanation_store.py tests/test_retrain.py tests/test_monitoring.py tests/test_ablation.py
api:
	uvicorn api.main:app --port 8000        # http://127.0.0.1:8000/docs
docker-api:
	docker build -f Dockerfile.api -t cfpb-risk-api .
dashboard:
	streamlit run app/dashboard.py          # http://localhost:8501
dashboard-data:
	python -m src.dashboard.build_assets    # rebuild app/assets/dashboard_data.json from the Module 4 outputs
dashboard-test:
	ruff check app monitoring_app src/dashboard scripts tests/test_dashboard.py tests/test_monitoring_app.py
	pytest -q tests/test_dashboard.py tests/test_monitoring_app.py
api-demo:
	python scripts/api_demo.py --spawn
docker-dashboard:
	docker build -f Dockerfile.dashboard -t cfpb-dashboard .
monitor:                                  # stage 1 in the model environment, stage 2 (Evidently) in the pandas-2 environment
	MODEL_PY=$${MODEL_PY:-python} MON_PY=$${MON_PY:-python} scripts/run_monitoring.sh
monitoring-dashboard:
	streamlit run monitoring_app/dashboard.py --server.port 8502
retrain-dry-run:                          # builds a challenger on historical windows, gates it, registers it as `challenger`; can never be promoted
	python -m src.modeling.retrain challenger --dry-run
compose-config:
	PIPELINE_PSEUDONYM_KEY=x docker compose --profile pipeline --profile jobs config -q
