from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    logger.warning("Fichier introuvable : %s -> DataFrame vide", path)
    return pd.DataFrame()


def build_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)
    return report


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    mon = cfg["monitoring"]

    reference = load_or_empty(Path(mon["reference_data"]))
    current = load_or_empty(Path(mon["current_data"]))

    if reference.empty or current.empty:
        raise RuntimeError(
            "Données de référence ou courantes manquantes. "
            "Lancez d'abord l'entraînement (référence) et accumulez des requêtes "
            "de production (courant) avant de générer un rapport."
        )

    report = build_report(reference, current)

    out_path = Path(mon["report_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))
    logger.info("Rapport de dérive enregistré : %s", out_path)

    result = report.as_dict()
    drift_summary = {
        "dataset_drift": result["metrics"][0]["result"].get("dataset_drift"),
        "drift_share": result["metrics"][0]["result"].get("drift_share"),
    }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(drift_summary, indent=2), encoding="utf-8")

    if drift_summary.get("drift_share", 0) and drift_summary["drift_share"] > mon["drift_share_threshold"]:
        logger.warning(
            "ALERTE : dérive détectée (%.0f%% > seuil %.0f%%) — un ré-entraînement est recommandé.",
            drift_summary["drift_share"] * 100,
            mon["drift_share_threshold"] * 100,
        )

    return drift_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rapport de dérive Evidently AI")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
