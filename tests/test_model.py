import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.features.build_features import fit_preprocessor, transform
from src.models.predict import PricePredictor
from src.models.train import evaluate


@pytest.fixture()
def synthetic_features_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "surface_reelle_bati": rng.uniform(20, 120, n),
            "nombre_pieces_principales": rng.integers(1, 6, n),
            "surface_terrain": np.zeros(n),
            "type_local": rng.choice(["Maison", "Appartement"], n),
            "code_departement": rng.choice(["75", "92"], n),
            "annee_mutation": rng.integers(2021, 2024, n),
            "mois_mutation": rng.integers(1, 12, n),
        }
    )
    # prix synthétique corrélé à la surface, pour vérifier que le modèle apprend un signal
    df["valeur_fonciere"] = df["surface_reelle_bati"] * 6000 + rng.normal(0, 5000, n)
    return df


def test_train_and_evaluate_learns_signal(synthetic_features_df, tmp_path):
    df = synthetic_features_df
    train_df, test_df = df.iloc[:160], df.iloc[160:]

    encoder, feature_names = fit_preprocessor(train_df)
    X_train = transform(train_df, encoder)
    X_test = transform(test_df, encoder)
    y_train, y_test = train_df["valeur_fonciere"], test_df["valeur_fonciere"]

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    assert metrics["r2"] > 0.5  # le modèle doit capter le signal surface -> prix

    # sauvegarde et rechargement, comme le ferait train.py / predict.py
    model_path = tmp_path / "model.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"
    joblib.dump(model, model_path)
    joblib.dump({"encoder": encoder, "feature_names": feature_names}, preprocessor_path)

    predictor = PricePredictor(str(model_path), str(preprocessor_path))
    prix = predictor.predict(
        surface_reelle_bati=60,
        type_local="Appartement",
        code_postal="75015",
        nombre_pieces_principales=3,
    )
    assert prix > 0
