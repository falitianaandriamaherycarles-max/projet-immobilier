.PHONY: install ingest preprocess features train evaluate pipeline api test lint \
        docker-build docker-up docker-down mlflow-ui monitor clean

install:
	pip install -r requirements.txt

ingest:
	python -m src.data.ingest --config configs/config.yaml

preprocess:
	python -m src.data.preprocess --config configs/config.yaml

features:
	python -m src.features.build_features --config configs/config.yaml

train:
	python -m src.models.train --config configs/config.yaml

evaluate:
	python -m src.models.evaluate --config configs/config.yaml

# Pipeline complet, de l'ingestion à l'évaluation (équivalent à `dvc repro`)
pipeline: ingest preprocess features train evaluate

api:
	uvicorn src.api.main:app --reload --port 8000

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

monitor:
	python -m src.monitoring.drift_report --config configs/config.yaml

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src tests
	ruff format --check src tests

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
