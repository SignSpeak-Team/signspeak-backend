"""Endpoints para predicciones de lenguaje de señas"""

from fastapi import APIRouter, Depends, HTTPException
from api.models.request import LandmarksRequest, SequenceRequest
from api.models.response import PredictionResponse, ErrorResponse
from core.predictor import SignPredictor, get_predictor
import numpy as np

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "/static",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Predict Static Letter"
)
async def predict_static(
    request: LandmarksRequest,
    predictor: SignPredictor = Depends(get_predictor)
):
    """
    Predice una letra estática desde landmarks de mano.
    
    Args:
        request: LandmarksRequest con 21 landmarks (x, y, z cada uno)
    
    Returns:
        PredictionResponse con letra, confianza y tiempo de procesamiento
    
    Raises:
        HTTPException 400: Si el formato de landmarks es inválido
    """
    try:
        # Convertir landmarks a formato numpy (flatten to 63 features)
        landmarks_array = np.array(request.landmarks).flatten()
        
        # Predicción
        result = predictor.predict_static(landmarks_array)
        
        return PredictionResponse(**result)
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid landmarks format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@router.post(
    "/dynamic",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Predict Dynamic Letter"
)
async def predict_dynamic(
    request: SequenceRequest,
    predictor: SignPredictor = Depends(get_predictor)
):
    """
    Predice una letra dinámica desde secuencia de frames.
    
    Args:
        request: SequenceRequest con secuencia de 15 frames, cada uno con 21 landmarks
    
    Returns:
        PredictionResponse con letra, confianza y tiempo de procesamiento
    
    Raises:
        HTTPException 400: Si el formato de secuencia es inválido
    """
    try:
        # Convertir secuencia a formato numpy (15, 63)
        sequence_list = []
        for frame in request.sequence:
            frame_array = np.array(frame).flatten()
            sequence_list.append(frame_array)
        
        sequence_array = np.array(sequence_list)
        
        # Predicción
        result = predictor.predict_dynamic(sequence_array)
        
        return PredictionResponse(**result)
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sequence format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
