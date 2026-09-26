import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor

from src.features.build_features import fit_preprocessor, transform


@pytest.fixture()
def trained_model_files(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "surface_reelle_bati": rng.uniform(20, 100, n),
            "nombre_pieces_principales": rng.integers(1, 5, n),
            "surface_terrain": np.zeros(n),
            "type_local": rng.choice(["Maison", "Appartement"], n),
            "code_departement": rng.choice(["75", "92"], n),
            "annee_mutation": 2023,
            "mois_mutation": 6,
        }
    )
    df["valeur_fonciere"] = df["surface_reelle_bati"] * 6000

    encoder, feature_names = fit_preprocessor(df)
    X = transform(df, encoder)
    model = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, df["valeur_fonciere"])

    model_path = tmp_path / "model.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"
    joblib.dump(model, model_path)
    joblib.dump({"encoder": encoder, "feature_names": feature_names}, preprocessor_path)

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("PREPROCESSOR_PATH", str(preprocessor_path))

    # le cache lru_cache doit être vidé pour prendre en compte les nouveaux chemins
    from src.api.model_loader import get_predictor

    get_predictor.cache_clear()
    yield
    get_predictor.cache_clear()


@pytest.fixture()
def client(trained_model_files):
    from src.api.main import app

    return TestClient(app)


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_predict_valid_payload(client):
    payload = {
        "surface_reelle_bati": 55,
        "type_local": "Appartement",
        "code_postal": "75015",
        "nombre_pieces_principales": 3,
        "surface_terrain": 0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prix_estime"] > 0
    assert body["devise"] == "EUR"


def test_predict_rejects_invalid_type_local(client):
    payload = {
        "surface_reelle_bati": 55,
        "type_local": "Château",
        "code_postal": "75015",
        "nombre_pieces_principales": 3,
        "surface_terrain": 0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_bad_surface(client):
    payload = {
        "surface_reelle_bati": -10,
        "type_local": "Appartement",
        "code_postal": "75015",
        "nombre_pieces_principales": 3,
        "surface_terrain": 0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
