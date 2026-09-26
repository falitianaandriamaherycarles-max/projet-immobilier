import pandas as pd
import pytest

from src.data.preprocess import clean_dvf

CFG = {
    "preprocess": {
        "types_locaux_valides": ["Maison", "Appartement"],
        "nature_mutation_valide": "Vente",
        "prix_min": 10000,
        "prix_max": 3000000,
        "surface_min": 9,
        "surface_max": 500,
        "prix_m2_min": 500,
        "prix_m2_max": 20000,
        "test_size": 0.2,
        "random_state": 42,
    }
}


def make_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # ligne valide
            dict(
                date_mutation="2023-05-10",
                nature_mutation="Vente",
                valeur_fonciere=250000,
                code_postal="75015",
                type_local="Appartement",
                surface_reelle_bati=50,
                nombre_pieces_principales=2,
                surface_terrain=None,
            ),
            # mauvaise nature de mutation -> rejetée
            dict(
                date_mutation="2023-05-10",
                nature_mutation="Échange",
                valeur_fonciere=250000,
                code_postal="75015",
                type_local="Appartement",
                surface_reelle_bati=50,
                nombre_pieces_principales=2,
                surface_terrain=None,
            ),
            # type local non retenu -> rejetée
            dict(
                date_mutation="2023-05-10",
                nature_mutation="Vente",
                valeur_fonciere=250000,
                code_postal="75015",
                type_local="Local industriel",
                surface_reelle_bati=50,
                nombre_pieces_principales=2,
                surface_terrain=None,
            ),
            # prix aberrant -> rejetée
            dict(
                date_mutation="2023-05-10",
                nature_mutation="Vente",
                valeur_fonciere=1,
                code_postal="75015",
                type_local="Maison",
                surface_reelle_bati=50,
                nombre_pieces_principales=4,
                surface_terrain=200,
            ),
        ]
    )


def test_clean_dvf_filters_correctly():
    raw = make_raw_df()
    clean = clean_dvf(raw, CFG)

    assert len(clean) == 1
    assert clean.iloc[0]["type_local"] == "Appartement"
    assert "prix_m2" in clean.columns
    assert clean.iloc[0]["prix_m2"] == pytest.approx(250000 / 50)


def test_clean_dvf_postal_code_is_zero_padded():
    raw = make_raw_df().iloc[[0]].copy()
    raw["code_postal"] = 7501  # simulateur d'un code postal mal typé
    clean = clean_dvf(raw, CFG)
    assert clean.iloc[0]["code_postal"] == "07501"
