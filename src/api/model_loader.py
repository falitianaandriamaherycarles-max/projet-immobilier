from __future__ import annotations

import os
from functools import lru_cache

from src.models.predict import PricePredictor

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")
PREPROCESSOR_PATH = os.environ.get("PREPROCESSOR_PATH", "models/preprocessor.pkl")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "local")


@lru_cache(maxsize=1)
def get_predictor() -> PricePredictor:
    return PricePredictor(MODEL_PATH, PREPROCESSOR_PATH)
