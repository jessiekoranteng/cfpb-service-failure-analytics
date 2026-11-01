.PHONY: lint test pipeline docker
lint:
	ruff check src flows tests
test: lint
	pytest -q
pipeline:
	PIPELINE_PSEUDONYM_KEY=$${PIPELINE_PSEUDONYM_KEY:-dev-key-not-for-prod} python -m flows.ccdb_pipeline --source data/raw/complaints.csv.zip
docker:
	docker build -t cfpb-pipeline .
