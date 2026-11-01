.PHONY: lint test pipeline docker model-lint model-test api docker-api
lint:
	ruff check src flows tests
test: lint
	pytest -q --ignore=tests/test_modeling.py
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
