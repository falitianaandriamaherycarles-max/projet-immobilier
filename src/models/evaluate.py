from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET = "valeur_fonciere"


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main(config_path: str = "configs/config.yaml") -> dict:
    cfg = load_config(config_path)
    features_dir = Path(cfg["data"]["features_dir"])

    test = pd.read_parquet(features_dir / "test.parquet")
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    model = joblib.load(cfg["train"]["model_path"])
    preds = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)

    report_path = Path("reports/eval_metrics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Métriques : %s", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation du modèle")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
