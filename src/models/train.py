from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET = "valeur_fonciere"


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(features_dir: Path):
    train = pd.read_parquet(features_dir / "train.parquet")
    test = pd.read_parquet(features_dir / "test.parquet")
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    return X_train, y_train, X_test, y_test


def build_model(params: dict) -> RandomForestRegressor:
    return RandomForestRegressor(**params)


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": root_mean_squared_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }


def main(config_path: str = "configs/config.yaml"):
    cfg = load_yaml(config_path)
    model_params = load_yaml(cfg["train"]["params_file"])["random_forest"]

    features_dir = Path(cfg["data"]["features_dir"])
    X_train, y_train, X_test, y_test = load_split(features_dir)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params(model_params)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        model = build_model(model_params)
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        logger.info("Métriques test : %s", metrics)

        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=cfg["mlflow"]["registered_model_name"],
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # Artefact local, versionné par DVC (dvc.yaml -> stage "train")
        model_path = Path(cfg["train"]["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info("Modèle enregistré localement : %s", model_path)

        mlflow.log_artifact(str(model_path))

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement du modèle")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
