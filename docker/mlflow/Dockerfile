FROM python:3.11-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow>=3.5.0,<4

EXPOSE 5000

# Backend store SQLite local + artefacts sur volume monté (voir docker-compose.yml)
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts"]
