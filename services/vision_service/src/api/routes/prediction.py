"""Endpoints para predicciones de lenguaje de señas"""

from fastapi import APIRouter, Depends, HTTPException
from api.models.request import LandmarksRequest, SequenceRequest
from api.models.response import (
    PredictionResponse, ErrorResponse, 
    WordPredictionResponse, BufferStatsResponse
)
from core.predictor import SignPredictor, get_predictor
import numpy as np
import time

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


@router.post(
    "/words",
    response_model=WordPredictionResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Predict Word with Buffer"
)
async def predict_words(
    request: SequenceRequest,
    predictor: SignPredictor = Depends(get_predictor)
):
    """
    Predice una palabra desde secuencia de frames con filtrado de buffer.
    
    El buffer filtra repeticiones (cooldown 2s) y baja confianza (<80%).
    Las palabras aceptadas se acumulan en una frase.
    
    Args:
        request: SequenceRequest con secuencia de 15 frames
    
    Returns:
        WordPredictionResponse con palabra, confianza, frase acumulada
    """
    try:
        start_time = time.time()
        
        # Convertir secuencia a formato numpy
        sequence_list = []
        for frame in request.sequence:
            frame_array = np.array(frame).flatten()
            sequence_list.append(frame_array)
        
        # Alimentar cada frame al buffer del predictor
        result = None
        for frame_landmarks in sequence_list:
            result = predictor.predict_word_with_buffer(frame_landmarks)
        
        processing_time = (time.time() - start_time) * 1000
        
        if result is None:
            # No hay suficientes frames o fue filtrado
            return WordPredictionResponse(
                word="",
                confidence=0.0,
                phrase=predictor.get_current_phrase(),
                accepted=False,
                processing_time_ms=round(processing_time, 2)
            )
        
        return WordPredictionResponse(
            word=result["word"],
            confidence=round(result["confidence"], 2),
            phrase=result.get("phrase", predictor.get_current_phrase()),
            accepted=result.get("accepted", True),
            processing_time_ms=round(processing_time, 2)
        )
    
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


@router.get(
    "/words/stats",
    response_model=BufferStatsResponse,
    summary="Get Word Buffer Statistics"
)
async def get_word_buffer_stats(
    predictor: SignPredictor = Depends(get_predictor)
):
    """Retorna estadísticas del buffer de palabras."""
    stats = predictor.get_word_buffer_stats()
    return BufferStatsResponse(**stats)


@router.post(
    "/words/clear",
    summary="Clear Word Buffer"
)
async def clear_word_buffer(
    predictor: SignPredictor = Depends(get_predictor)
):
    """Limpia el buffer de palabras y reinicia la frase."""
    predictor.reset_buffer("word_buffer")
    return {"message": "Word buffer cleared", "phrase": ""}

