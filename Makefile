.PHONY: lint test pipeline docker model-lint model-test api docker-api dashboard dashboard-test docker-dashboard api-demo dashboard-data
lint:
	ruff check src flows tests
test: lint
	pytest -q --ignore=tests/test_modeling.py --ignore=tests/test_dashboard.py
pipeline:
	PIPELINE_PSEUDONYM_KEY=$${PIPELINE_PSEUDONYM_KEY:-dev-key-not-for-prod} python -m flows.ccdb_pipeline --source data/raw/complaints.csv.zip
docker:
	docker build -t cfpb-pipeline .
model-lint:
	ruff check src/modeling api tests/test_modeling.py
model-test: model-lint
	pytest -q tests/test_modeling.py
api:
	uvicorn api.main:app --port 8000        # http://127.0.0.1:8000/docs
docker-api:
	docker build -f Dockerfile.api -t cfpb-risk-api .
dashboard:
	streamlit run app/dashboard.py          # http://localhost:8501
dashboard-data:
	python -m src.dashboard.build_assets    # rebuild app/assets/dashboard_data.json from the Module 4 outputs
dashboard-test:
	ruff check app src/dashboard scripts tests/test_dashboard.py
	pytest -q tests/test_dashboard.py
api-demo:
	python scripts/api_demo.py --spawn
docker-dashboard:
	docker build -f Dockerfile.dashboard -t cfpb-dashboard .
