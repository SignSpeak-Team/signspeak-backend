"""Prediction Routes - Proxy endpoints to Vision Service."""

from fastapi import APIRouter, HTTPException
from src.schemas.prediction import (
    BufferStatsResponse,
    HolisticRequest,
    LandmarksRequest,
    LetterPredictionResponse,
    SequenceRequest,
    WordPredictionResponse,
)
from src.services.vision_client import (
    VisionServiceError,
    clear_holistic_buffer,
    clear_word_buffer,
    get_word_buffer_stats,
    predict_dynamic,
    predict_holistic,
    predict_static,
    predict_words,
)

router = APIRouter(
    prefix="/api/v1/predict",
    tags=["Prediction"],
)


# === Letter Predictions ===


@router.post(
    "/static",
    response_model=LetterPredictionResponse,
    summary="Predict Static Letter",
    description="Predict a static letter (A-Z except J, K, Q, X, Z, Ñ) from 21 hand landmarks.",
)
async def predict_static_letter(request: LandmarksRequest):
    """
    Predict static letter from hand landmarks.

    - **landmarks**: 21 hand landmarks, each with [x, y, z] normalized coordinates
    """
    try:
        result = await predict_static(request.landmarks)
        return LetterPredictionResponse(**result)
    except VisionServiceError as e:
        raise HTTPException(status_code=e.status_code or 502, detail=e.message) from e


@router.post(
    "/dynamic",
    response_model=LetterPredictionResponse,
    summary="Predict Dynamic Letter",
    description="Predict a dynamic letter (J, K, Q, X, Z, Ñ) from 15 frames of landmarks.",
)
async def predict_dynamic_letter(request: SequenceRequest):
    """
    Predict dynamic letter from sequence of frames.

    - **sequence**: 15 frames × 21 landmarks × 3 coordinates
    """
    try:
        result = await predict_dynamic(request.sequence)
        return LetterPredictionResponse(**result)
    except VisionServiceError as e:
        raise HTTPException(status_code=e.status_code or 502, detail=e.message) from e


# === Word Predictions ===


@router.post(
    "/words",
    response_model=WordPredictionResponse,
    summary="Predict Word",
    description="Predict a word from 249-word LSM vocabulary using 15 frames of landmarks.",
)
async def predict_word(request: SequenceRequest):
    """
    Predict word from sequence of frames.

    Uses internal buffer for smoothing and phrase accumulation.

    - **sequence**: 15 frames × 21 landmarks × 3 coordinates
    """
    try:
        result = await predict_words(request.sequence)
        return WordPredictionResponse(**result)
    except VisionServiceError as e:
        raise HTTPException(status_code=e.status_code or 502, detail=e.message) from e


@router.post(
    "/holistic",
    response_model=WordPredictionResponse,
    summary="Predict Medical Word",
    description="Predict a medical vocabulary word (150 words) from holistic body features.",
)
async def predict_medical_word(request: HolisticRequest):
    """
    Predict medical word from holistic features.

    - **landmarks**: 226 features (pose 33×4 + left hand 21×3 + right hand 21×3)
    """
    try:
        result = await predict_holistic(request.landmarks)
        return WordPredictionResponse(**result)
    except VisionServiceError as e:
        raise HTTPException(status_code=e.status_code or 502, detail=e.message) from e


# === Buffer Management ===


@router.get(
    "/words/stats",
    response_model=BufferStatsResponse,
    summary="Get Buffer Statistics",
)
async def get_buffer_stats():
    """Get word buffer and phrase statistics."""
    try:
        result = await get_word_buffer_stats()
        return BufferStatsResponse(**result)
    except VisionServiceError as e:
        raise HTTPException(status_code=e.status_code or 502, detail=e.message) from e


@router.post("/words/clear", summary="Clear Word Buffer")
async def clear_words():
    """Clear word prediction buffer and accumulated phrase."""
    try:
        return await clear_word_buffer()
    except VisionServiceError as e:
        raise HTTPException(status_code=e.status_code or 502, detail=e.message) from e


@router.post("/holistic/clear", summary="Clear Holistic Buffer")
async def clear_holistic():
    """Clear holistic prediction buffer."""
    try:
        return await clear_holistic_buffer()
    except VisionServiceError as e:
        raise HTTPException(status_code=e.status_code or 502, detail=e.message) from e
