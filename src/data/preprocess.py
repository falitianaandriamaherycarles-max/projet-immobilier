from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_dvf(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pp = cfg["preprocess"]
    n0 = len(df)

    df = df.copy()

    # 1. Ne garder que les ventes de type Maison / Appartement
    df = df[df["nature_mutation"] == pp["nature_mutation_valide"]]
    df = df[df["type_local"].isin(pp["types_locaux_valides"])]

    # 2. Colonnes essentielles non nulles
    required = ["valeur_fonciere", "surface_reelle_bati", "type_local", "code_postal"]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # 3. Bornes de prix et de surface (retrait des valeurs aberrantes)
    df = df[df["valeur_fonciere"].between(pp["prix_min"], pp["prix_max"])]
    df = df[df["surface_reelle_bati"].between(pp["surface_min"], pp["surface_max"])]

    # 4. Prix au m² et filtrage des outliers résiduels
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    df = df[df["prix_m2"].between(pp["prix_m2_min"], pp["prix_m2_max"])]

    # 5. Doublons exacts
    df = df.drop_duplicates()

    # 6. Types
    df["code_postal"] = df["code_postal"].astype(str).str.zfill(5)
    if "nombre_pieces_principales" in df.columns:
        df["nombre_pieces_principales"] = df["nombre_pieces_principales"].fillna(0).astype(int)
    if "surface_terrain" in df.columns:
        df["surface_terrain"] = df["surface_terrain"].fillna(0)

    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
    df = df.dropna(subset=["date_mutation"])

    logger.info("Nettoyage : %d -> %d lignes (%.1f%% conservées)", n0, len(df), 100 * len(df) / n0 if n0 else 0)
    return df.reset_index(drop=True)


def main(config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    dcfg = cfg["data"]

    raw_path = Path(dcfg["raw_dir"]) / "dvf_raw.csv"
    df = pd.read_csv(raw_path, low_memory=False)

    clean = clean_dvf(df, cfg)

    out_dir = Path(dcfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dvf_clean.parquet"
    clean.to_parquet(out_path, index=False)
    logger.info("Données nettoyées enregistrées : %s", out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prétraitement des données DVF")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
