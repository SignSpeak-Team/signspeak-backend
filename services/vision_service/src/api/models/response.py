"""API Response Models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Letter prediction response."""

    letter: str
    confidence: float
    type: Literal["static", "dynamic"]
    processing_time_ms: float


class WordPredictionResponse(BaseModel):
    """Word prediction response."""

    word: str
    confidence: float
    phrase: str | None = ""
    accepted: bool = True
    processing_time_ms: float = 0
    type: str | None = None


class HealthResponse(BaseModel):
    """Service health check."""

    status: Literal["healthy", "unhealthy"]
    version: str
    models_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.now)


class BufferStatsResponse(BaseModel):
    """Word buffer statistics."""

    total_received: int
    total_accepted: int
    rejected_by_cooldown: int
    rejected_by_confidence: int
    acceptance_rate: float
    current_phrase: str


class ModelsInfoResponse(BaseModel):
    """Models information."""

    static: dict
    dynamic: dict
    words: dict
    holistic: dict


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str
    error_code: str = "ERROR"
