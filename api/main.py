"""FastAPI inference service for the selected ESM-2 localization model."""

from __future__ import annotations

import os
import sys
import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from esm2_localization.inference import ProteinPredictor

DEFAULT_CHECKPOINT = PROJECT_ROOT / "results/models/esm2_150m/best_checkpoint.pt"
LOGGER = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    protein_id: str = Field(default="query_protein", min_length=1, max_length=200)
    sequence: str = Field(min_length=1, max_length=100_000)
    window_size: int = Field(default=1022, ge=1, le=1022)
    window_stride: int = Field(default=894, ge=1, le=1022)
    window_batch_size: int = Field(default=4, ge=1, le=64)


class WindowResult(BaseModel):
    window_index: int
    start: int
    end: int
    membrane_probability: float


class PredictionResponse(BaseModel):
    protein_id: str
    sequence_length: int
    predicted_label: Literal["membrane", "soluble"]
    membrane_probability: float
    soluble_probability: float
    threshold: float
    model_name: str
    checkpoint_epoch: int | None
    device: str
    long_sequence_mode: bool
    aggregation: str
    window_size: int
    window_stride: int
    number_of_windows: int
    top_window: WindowResult
    windows: list[WindowResult]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool
    checkpoint_configured: bool


@lru_cache(maxsize=1)
def get_predictor() -> ProteinPredictor:
    checkpoint = Path(os.getenv("MODEL_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    device = os.getenv("MODEL_DEVICE", "auto")
    return ProteinPredictor(checkpoint, device=device)


def require_predictor() -> ProteinPredictor:
    """Translate model-startup failures into an actionable API response."""
    try:
        return get_predictor()
    except Exception as error:
        LOGGER.exception("Model initialization failed")
        raise HTTPException(
            status_code=503,
            detail=(
                "The model could not be loaded. Check MODEL_CHECKPOINT, confirm "
                "that it points to a real .pt file, and inspect the service terminal."
            ),
        ) from error


app = FastAPI(
    title="ESM-2 Protein Localization API",
    version="2.0.0",
    description=(
        "Predict binary membrane-associated versus soluble localization from "
        "one amino-acid sequence. Long proteins use validated overlapping windows."
    ),
)
WEB_DIR = PROJECT_ROOT / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", tags=["service"])
def service_info() -> dict:
    return {
        "service": app.title,
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "prediction": "/v1/predict",
    }


@app.get("/demo", include_in_schema=False)
def demo() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    checkpoint = Path(os.getenv("MODEL_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    return HealthResponse(
        model_loaded=get_predictor.cache_info().currsize > 0,
        checkpoint_configured=checkpoint.exists(),
    )


@app.post("/v1/predict", response_model=PredictionResponse, tags=["inference"])
def predict(
    request: PredictionRequest,
    predictor: Annotated[ProteinPredictor, Depends(require_predictor)],
) -> PredictionResponse:
    try:
        result = predictor.predict(
            sequence=request.sequence,
            protein_id=request.protein_id,
            window_size=request.window_size,
            window_stride=request.window_stride,
            window_batch_size=request.window_batch_size,
        )
        # Direct construction works with both Pydantic v1 and v2. The
        # model_validate() helper is available only in v2.
        return PredictionResponse(**result)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
