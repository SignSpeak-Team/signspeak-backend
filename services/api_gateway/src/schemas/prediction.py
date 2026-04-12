"""Prediction Schemas - Request/Response models for Vision Service predictions."""

from pydantic import BaseModel, Field

# === Request Models ===


class LandmarksRequest(BaseModel):
    """Request for static letter prediction (21 hand landmarks)."""

    landmarks: list[list[float]] = Field(
        ...,
        min_length=21,
        max_length=21,
        description="21 hand landmarks, each with [x, y, z] coordinates (0.0-1.0)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "landmarks": [[0.5, 0.5, 0.0] for _ in range(21)],
            }
        }


class SequenceRequest(BaseModel):
    """Request for dynamic letter or word prediction (sequence of frames)."""

    sequence: list[list[list[float]]] = Field(
        ...,
        min_length=15,
        max_length=15,
        description="15 frames, each with 21 landmarks × 3 coordinates",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sequence": [[[0.5, 0.5, 0.0] for _ in range(21)] for _ in range(15)],
            }
        }


class HolisticRequest(BaseModel):
    """Request for holistic (medical vocabulary) prediction."""

    landmarks: list[float] = Field(
        ...,
        min_length=226,
        max_length=226,
        description="226 holistic features: pose (33×4) + left hand (21×3) + right hand (21×3)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "landmarks": [0.5] * 226,
            }
        }


# === Response Models ===


class LetterPredictionResponse(BaseModel):
    """Response for letter prediction (static or dynamic)."""

    letter: str = Field(..., description="Predicted letter (A-Z, Ñ)")
    confidence: float = Field(
        ..., ge=0.0, le=100.0, description="Confidence percentage (0-100)"
    )
    type: str = Field(..., description="Prediction type: 'static' or 'dynamic'")
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds"
    )


class WordPredictionResponse(BaseModel):
    """Response for word prediction."""

    word: str = Field(..., description="Predicted word")
    confidence: float = Field(
        ..., ge=0.0, le=100.0, description="Confidence percentage (0-100)"
    )
    phrase: str = Field(default="", description="Accumulated phrase from predictions")
    accepted: bool = Field(
        default=False, description="Whether the prediction was accepted to the phrase"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds"
    )


class BufferStatsResponse(BaseModel):
    """Response for buffer statistics."""

    total_received: int = Field(..., description="Total received predictions")
    total_accepted: int = Field(..., description="Total accepted predictions")
    rejected_by_cooldown: int = Field(..., description="Rejected by cooldown")
    rejected_by_confidence: int = Field(..., description="Rejected by low confidence")
    acceptance_rate: float = Field(..., description="Acceptance rate percentage")
    current_phrase: str = Field(..., description="Current accumulated phrase")
