"""Prediction Endpoints."""

from fastapi import APIRouter, Depends, HTTPException
import numpy as np

from api.models.request import LandmarksRequest, SequenceRequest, HolisticRequest
from api.models.response import PredictionResponse, WordPredictionResponse, BufferStatsResponse
from core.predictor import SignPredictor, get_predictor

router = APIRouter(prefix="/predict", tags=["Prediction"])


# === Letter Predictions ===

@router.post("/static", response_model=PredictionResponse)
async def predict_static(
    request: LandmarksRequest,
    predictor: SignPredictor = Depends(get_predictor)
):
    """Predict static letter (21 landmarks → 63 features)."""
    try:
        landmarks = np.array(request.landmarks).flatten()
        return PredictionResponse(**predictor.predict_static(landmarks))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/dynamic", response_model=PredictionResponse)
async def predict_dynamic(
    request: SequenceRequest,
    predictor: SignPredictor = Depends(get_predictor)
):
    """Predict dynamic letter (15 frames × 63 features)."""
    try:
        sequence = np.array([[np.array(lm).flatten() for lm in frame] 
                            for frame in request.sequence])
        sequence = sequence.reshape(15, 63)
        return PredictionResponse(**predictor.predict_dynamic(sequence))
    except ValueError as e:
        raise HTTPException(400, str(e))


# === Word Predictions ===

@router.post("/words", response_model=WordPredictionResponse)
async def predict_words(
    request: SequenceRequest,
    predictor: SignPredictor = Depends(get_predictor)
):
    """Predict word with buffer filtering (15 frames × 63 features)."""
    try:
        result = None
        for frame in request.sequence:
            landmarks = np.array(frame).flatten()
            result = predictor.predict_word_with_buffer(landmarks)
        
        if result is None:
            return WordPredictionResponse(
                word="", confidence=0.0,
                phrase=predictor.get_current_phrase(),
                accepted=False, processing_time_ms=0
            )
        return WordPredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/holistic", response_model=WordPredictionResponse)
async def predict_holistic(
    request: HolisticRequest,
    predictor: SignPredictor = Depends(get_predictor)
):
    """Predict medical word (226 holistic features: pose + hands)."""
    try:
        landmarks = np.array(request.landmarks)
        result = predictor.predict_holistic(landmarks)
        
        if result is None:
            return WordPredictionResponse(
                word="", confidence=0.0, phrase="",
                accepted=False, processing_time_ms=0
            )
        return WordPredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(400, str(e))


# === Buffer Management ===

@router.get("/words/stats", response_model=BufferStatsResponse)
async def get_word_buffer_stats(predictor: SignPredictor = Depends(get_predictor)):
    """Get word buffer statistics."""
    return BufferStatsResponse(**predictor.get_word_buffer_stats())


@router.post("/words/clear")
async def clear_word_buffer(predictor: SignPredictor = Depends(get_predictor)):
    """Clear word buffer and phrase."""
    predictor.reset_buffer("word_buffer")
    return {"message": "Buffer cleared"}


@router.post("/holistic/clear")
async def clear_holistic_buffer(predictor: SignPredictor = Depends(get_predictor)):
    """Clear holistic buffer."""
    predictor.reset_buffer("holistic")
    return {"message": "Holistic buffer cleared"}
