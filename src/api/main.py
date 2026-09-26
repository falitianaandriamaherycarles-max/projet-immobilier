from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from src.api.model_loader import MODEL_VERSION, get_predictor
from src.api.schemas import HealthResponse, PredictionRequest, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Prédiction Prix Immobilier",
    description="Prédit le prix d'un bien à partir de ses caractéristiques (données DVF).",
    version="1.0.0",
)

TYPES_VALIDES = {"Maison", "Appartement"}


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    try:
        get_predictor()
        loaded = True
    except Exception as exc:  # modèle absent / non entraîné
        logger.error("Modèle non chargé : %s", exc)
        loaded = False
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(req: PredictionRequest) -> PredictionResponse:
    if req.type_local not in TYPES_VALIDES:
        raise HTTPException(status_code=422, detail=f"type_local doit être parmi {TYPES_VALIDES}")

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Modèle indisponible, avez-vous lancé l'entraînement ? ({exc})",
        )

    prix = predictor.predict(
        surface_reelle_bati=req.surface_reelle_bati,
        type_local=req.type_local,
        code_postal=req.code_postal,
        nombre_pieces_principales=req.nombre_pieces_principales,
        surface_terrain=req.surface_terrain,
    )
    return PredictionResponse(prix_estime=round(prix, 2), model_version=MODEL_VERSION)


@app.get("/", tags=["monitoring"])
def root():
    return {"message": "API Prédiction Prix Immobilier — voir /docs pour la documentation interactive"}
