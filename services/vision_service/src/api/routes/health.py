"""Endpoints para health checks y información del servicio"""

from core.predictor import SignPredictor, get_predictor
from fastapi import APIRouter, Depends

from api.models.response import HealthResponse, ModelsInfoResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse, summary="Health Check")
async def health_check():
    """
    Verifica el estado del servicio.

    Returns:
        HealthResponse con status, version y estado de modelos
    """
    try:
        get_predictor()
        models_loaded = True
    except Exception:
        models_loaded = False

    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        version="1.0.0",
        models_loaded=models_loaded,
    )


@router.get("/models", response_model=ModelsInfoResponse, summary="Models Information")
async def models_info(predictor: SignPredictor = Depends(get_predictor)):
    """
    Retorna información sobre los modelos ML cargados.

    Returns:
        ModelsInfoResponse con info de modelos estático y dinámico
    """
    info = predictor.get_models_info()
    return ModelsInfoResponse(**info)
