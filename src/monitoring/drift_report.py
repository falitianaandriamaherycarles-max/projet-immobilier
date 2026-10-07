from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset

try:
    from evidently.presets import DataSummaryPreset
except ImportError:
    DataSummaryPreset = None

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


def build_report(reference: pd.DataFrame, current: pd.DataFrame, drift_share: float = 0.5) -> Any:
    presets = [DataDriftPreset(drift_share=drift_share)]
    if DataSummaryPreset is not None:
        presets.append(DataSummaryPreset())
    return Report(presets).run(current_data=current, reference_data=reference)


def extract_drift_share(result: dict) -> float | None:
    for metric in result.get("metrics", []):
        if "DrifedColumnsCount" in json.dumps(metric, default=str):
            value = metric.get("value")
            if isinstance(value, dict) and "share" in value:
                return float(value["share"])
    return None


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    mon = cfg["monitoring"]
    threshold = mon["drift_share_threshold"]

    reference = load_or_empty(Path(mon["reference_data"]))
    current = load_or_empty(Path(mon["current_data"]))

    if reference.empty or current.empty:
        raise RuntimeError(
            "Données de référence ou courantes manquantes. "
            "Lancez d'abord l'entraînement (référence) et accumulez des requêtes "
            "de production (courant) avant de générer un rapport."
        )

    snapshot = build_report(reference, current, threshold)

    out_path = Path(mon["report_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_path))
    logger.info("Rapport de dérive enregistré : %s", out_path)

    drift_share = extract_drift_share(snapshot)
    if drift_share is None:
        logger.warning("Part de colonnes en derive introuvable dans le resultat d'Evidently.")
    drift_summary = {
        "dataset_drift": None if drift_share is None else drift_share >= threshold,
        "drift_share": drift_share,
    }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(drift_summary, indent=2), encoding="utf-8")

    if drift_share is not None and drift_share > threshold:
        logger.warning(
            "ALERTE : dérive détectée (%.0f%% > seuil %.0f%%) — un ré-entraînement est recommandé.",
            drift_share * 100,
            threshold * 100,
        )

    return drift_share


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rapport de dérive Evidently AI")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
