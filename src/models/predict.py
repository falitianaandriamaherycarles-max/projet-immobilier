from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

NUMERIC_FEATURES = ["surface_reelle_bati", "nombre_pieces_principales", "surface_terrain"]
CATEGORICAL_FEATURES = ["type_local", "code_departement"]


class PricePredictor:

    def __init__(self, model_path: str, preprocessor_path: str):
        self.model = joblib.load(model_path)
        bundle = joblib.load(preprocessor_path)
        self.encoder = bundle["encoder"]
        self.feature_names = bundle["feature_names"]

    def _to_frame(
        self,
        surface_reelle_bati: float,
        type_local: str,
        code_postal: str,
        nombre_pieces_principales: int = 0,
        surface_terrain: float = 0.0,
        annee_mutation: Optional[int] = None,
        mois_mutation: Optional[int] = None,
    ) -> pd.DataFrame:
        import datetime

        now = datetime.date.today()
        annee_mutation = annee_mutation or now.year
        mois_mutation = mois_mutation or now.month
        code_departement = str(code_postal)[:2]

        raw = pd.DataFrame(
            [
                {
                    "surface_reelle_bati": surface_reelle_bati,
                    "nombre_pieces_principales": nombre_pieces_principales,
                    "surface_terrain": surface_terrain,
                    "type_local": type_local,
                    "code_departement": code_departement,
                    "annee_mutation": annee_mutation,
                    "mois_mutation": mois_mutation,
                }
            ]
        )

        encoded = self.encoder.transform(raw[CATEGORICAL_FEATURES])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        numeric_df = raw[NUMERIC_FEATURES + ["annee_mutation", "mois_mutation"]]
        X = pd.concat([numeric_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # Réaligne les colonnes sur celles vues à l'entraînement (au cas où)
        X = X.reindex(columns=self.feature_names, fill_value=0)
        return X

    def predict(self, **kwargs) -> float:
        X = self._to_frame(**kwargs)
        return float(self.model.predict(X)[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prédiction unitaire en ligne de commande")
    parser.add_argument("--model", default="models/model.pkl")
    parser.add_argument("--preprocessor", default="models/preprocessor.pkl")
    parser.add_argument("--surface", type=float, required=True)
    parser.add_argument("--type-local", default="Appartement")
    parser.add_argument("--code-postal", default="75015")
    parser.add_argument("--pieces", type=int, default=3)
    args = parser.parse_args()

    predictor = PricePredictor(args.model, args.preprocessor)
    prix = predictor.predict(
        surface_reelle_bati=args.surface,
        type_local=args.type_local,
        code_postal=args.code_postal,
        nombre_pieces_principales=args.pieces,
    )
    print(f"Prix estimé : {prix:,.0f} €")
