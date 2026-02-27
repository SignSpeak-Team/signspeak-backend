"""Prediction Endpoints."""

import numpy as np
from core.predictor import SignPredictor, get_predictor
from fastapi import APIRouter, Depends, HTTPException

from api.models.request import (
    HolisticRequest,
    LSERequest,
    StaticLandmarksRequest,
    TemporalSequenceRequest,
)
from api.models.response import (
    BufferStatsResponse,
    PredictionResponse,
    WordPredictionResponse,
)

router = APIRouter(prefix="/predict", tags=["Prediction"])


# === Letter Predictions ===


@router.post("/static", response_model=PredictionResponse)
async def predict_static(
    request: StaticLandmarksRequest, predictor: SignPredictor = Depends(get_predictor)
):
    """
    Predict static letter from hand landmarks.

    **Input:** 21 landmarks × 3 coordinates (x, y, z normalized 0-1)

    **Vocabulary:** A, B, C, D, E, F, G, H, I, L, M, N, O, P, R, S, T, U, V, W, Y

    **Landmark Order (MediaPipe):**
    - 0: WRIST
    - 1-4: THUMB (CMC, MCP, IP, TIP)
    - 5-8: INDEX_FINGER
    - 9-12: MIDDLE_FINGER
    - 13-16: RING_FINGER
    - 17-20: PINKY
    """
    try:
        landmarks = np.array(request.landmarks).flatten()
        return PredictionResponse(**predictor.predict_static(landmarks))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/dynamic", response_model=PredictionResponse)
async def predict_dynamic(
    request: TemporalSequenceRequest, predictor: SignPredictor = Depends(get_predictor)
):
    """Predict dynamic letter (15 frames × 63 features)."""
    try:
        sequence = np.array(
            [[np.array(lm).flatten() for lm in frame] for frame in request.sequence]
        )
        sequence = sequence.reshape(15, 63)
        return PredictionResponse(**predictor.predict_dynamic(sequence))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


# === Word Predictions ===


@router.post("/words", response_model=WordPredictionResponse)
async def predict_words(
    request: TemporalSequenceRequest, predictor: SignPredictor = Depends(get_predictor)
):
    """Predict word with buffer filtering (15 frames × 63 features)."""
    try:
        result = None
        for frame in request.sequence:
            landmarks = np.array(frame).flatten()
            result = predictor.predict_word_with_buffer(landmarks)

        if result is None:
            return WordPredictionResponse(
                word="",
                confidence=0.0,
                phrase=predictor.get_current_phrase(),
                accepted=False,
                processing_time_ms=0,
            )
        return WordPredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/holistic", response_model=WordPredictionResponse)
async def predict_holistic(
    request: HolisticRequest, predictor: SignPredictor = Depends(get_predictor)
):
    """Predict medical word (226 holistic features: pose + hands)."""
    try:
        landmarks = np.array(request.landmarks)
        result = predictor.predict_holistic(landmarks)

        if result is None:
            return WordPredictionResponse(
                word="", confidence=0.0, phrase="", accepted=False, processing_time_ms=0
            )
        return WordPredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/lse", response_model=WordPredictionResponse)
async def predict_lse(
    request: LSERequest, predictor: SignPredictor = Depends(get_predictor)
):
    """Predict LSE sign (MSG3D: 75 landmarks)."""
    try:
        # Request.landmarks is list[list[float]] -> (75, 3)
        landmarks = np.array(request.landmarks)
        result = predictor.predict_lse(landmarks)

        if result is None:
            return WordPredictionResponse(
                word="", confidence=0.0, phrase="", accepted=False, processing_time_ms=0
            )
        return WordPredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


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


@router.post("/lse/clear")
async def clear_lse_buffer(predictor: SignPredictor = Depends(get_predictor)):
    """Clear LSE buffer."""
    predictor.reset_buffer("lse")
    return {"message": "LSE buffer cleared"}
