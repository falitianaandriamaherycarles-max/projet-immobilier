from __future__ import annotations

import argparse
import gzip
import logging
import shutil
from pathlib import Path

import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_file(url: str, dest: Path, timeout: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Téléchargement : %s", url)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    logger.info("Enregistré : %s (%.1f Mo)", dest, dest.stat().st_size / 1e6)
    return dest


def gunzip(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return dest


def fetch_departement_year(base_url: str, year: int, dept: str, raw_dir: Path) -> Path:
    gz_url = f"{base_url}/{year}/departements/{dept}.csv.gz"
    gz_path = raw_dir / f"{year}_{dept}.csv.gz"
    csv_path = raw_dir / f"{year}_{dept}.csv"

    if csv_path.exists():
        logger.info("Déjà présent, on saute : %s", csv_path)
        return csv_path

    try:
        download_file(gz_url, gz_path)
        gunzip(gz_path, csv_path)
        gz_path.unlink(missing_ok=True)
    except requests.HTTPError as exc:
        logger.error(
            "Échec du téléchargement pour %s/%s : %s. "
            "Vérifiez l'URL courante sur "
            "https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/",
            year,
            dept,
            exc,
        )
        raise
    return csv_path


def concatenate_raw(csv_paths: list[Path], columns_keep: list[str] | None, out_path: Path) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        logger.info("Lecture : %s", p)
        df = pd.read_csv(p, sep=",", low_memory=False)
        if columns_keep:
            available = [c for c in columns_keep if c in df.columns]
            df = df[available]
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)
    logger.info("Fichier brut consolidé : %s (%d lignes)", out_path, len(full))
    return full


def main(config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    dcfg = cfg["data"]
    raw_dir = Path(dcfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = []
    for year in dcfg["years"]:
        for dept in dcfg["departements"]:
            try:
                csv_paths.append(fetch_departement_year(dcfg["base_url"], year, dept, raw_dir))
            except Exception:
                logger.warning("On continue malgré l'échec pour %s/%s", year, dept)

    if not csv_paths:
        raise RuntimeError(
            "Aucun fichier DVF n'a pu être téléchargé. "
            "Vérifiez la connexion réseau et l'URL de configuration (data.base_url)."
        )

    out_path = raw_dir / "dvf_raw.csv"
    concatenate_raw(csv_paths, dcfg.get("columns_keep"), out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion des données DVF")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
