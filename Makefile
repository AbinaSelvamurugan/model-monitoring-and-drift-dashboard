.PHONY: prepare api worker dashboard test lint format docker
prepare:
	python -m modelwatch.cli prepare
api:
	uvicorn modelwatch.api:app --host 127.0.0.1 --port 8000
worker:
	python -m modelwatch.worker
dashboard:
	streamlit run modelwatch/dashboard.py
test:
	pytest -q
lint:
	ruff check .
format:
	ruff format .
docker:
	docker compose up --build -d
