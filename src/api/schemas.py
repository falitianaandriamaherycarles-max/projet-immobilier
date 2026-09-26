from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    surface_reelle_bati: float = Field(..., gt=0, le=1000, description="Surface habitable en m²")
    type_local: str = Field(..., description="'Maison' ou 'Appartement'")
    code_postal: str = Field(..., min_length=5, max_length=5, description="Code postal (5 chiffres)")
    nombre_pieces_principales: int = Field(3, ge=0, le=20)
    surface_terrain: float = Field(0.0, ge=0, le=100000)

    model_config = {
        "json_schema_extra": {
            "example": {
                "surface_reelle_bati": 65,
                "type_local": "Appartement",
                "code_postal": "75015",
                "nombre_pieces_principales": 3,
                "surface_terrain": 0,
            }
        }
    }


class PredictionResponse(BaseModel):
    prix_estime: float
    devise: str = "EUR"
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
