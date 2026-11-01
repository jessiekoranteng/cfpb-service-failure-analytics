.PHONY: test lint
lint:
	ruff check src tests
test: lint
	pytest -q
