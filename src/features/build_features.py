from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["surface_reelle_bati", "nombre_pieces_principales", "surface_terrain"]
CATEGORICAL_FEATURES = ["type_local", "code_departement"]
TARGET = "valeur_fonciere"


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["annee_mutation"] = df["date_mutation"].dt.year
    df["mois_mutation"] = df["date_mutation"].dt.month
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:

    df = add_calendar_features(df)
    if "code_departement" not in df.columns and "code_postal" in df.columns:
        df["code_departement"] = df["code_postal"].astype(str).str[:2]

    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["annee_mutation", "mois_mutation", TARGET]
    cols = [c for c in cols if c in df.columns]
    return df[cols].dropna()


def fit_preprocessor(df: pd.DataFrame) -> tuple[OneHotEncoder, list[str]]:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(df[CATEGORICAL_FEATURES])
    feature_names = (
        NUMERIC_FEATURES
        + ["annee_mutation", "mois_mutation"]
        + list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    )
    return encoder, feature_names


def transform(df: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    encoded = encoder.transform(df[CATEGORICAL_FEATURES])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES), index=df.index)
    numeric_df = df[NUMERIC_FEATURES + ["annee_mutation", "mois_mutation"]].reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    return pd.concat([numeric_df, encoded_df], axis=1)


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    dcfg = cfg["data"]
    pp = cfg["preprocess"]

    clean_path = Path(dcfg["processed_dir"]) / "dvf_clean.parquet"
    df = pd.read_parquet(clean_path)

    feat_df = build_feature_frame(df)

    train_df, test_df = train_test_split(
        feat_df, test_size=pp["test_size"], random_state=pp["random_state"]
    )

    encoder, feature_names = fit_preprocessor(train_df)

    X_train = transform(train_df, encoder)
    X_test = transform(test_df, encoder)
    y_train = train_df[TARGET].reset_index(drop=True)
    y_test = test_df[TARGET].reset_index(drop=True)

    out_dir = Path(dcfg["features_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.concat([X_train, y_train.rename(TARGET)], axis=1).to_parquet(out_dir / "train.parquet", index=False)
    pd.concat([X_test, y_test.rename(TARGET)], axis=1).to_parquet(out_dir / "test.parquet", index=False)

    models_dir = Path(cfg["train"]["model_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"encoder": encoder, "feature_names": feature_names}, cfg["train"]["preprocessor_path"])

    logger.info("Features enregistrées dans %s (train=%d, test=%d)", out_dir, len(X_train), len(X_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
