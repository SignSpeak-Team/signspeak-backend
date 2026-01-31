"""Vision Service Client - HTTP client for communication with Vision Service."""

from typing import Any

import httpx
from src.settings import settings


class VisionServiceError(Exception):
    """Custom exception for Vision Service errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


async def _make_request(
    method: str,
    endpoint: str,
    json_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Make HTTP request to Vision Service."""
    url = f"{settings.VISION_SERVICE_URL}/api/v1{endpoint}"
    timeout = httpx.Timeout(
        settings.HTTP_TIMEOUT,
        connect=settings.HTTP_CONNECT_TIMEOUT,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.request(method, url, json=json_data)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as e:
            raise VisionServiceError(
                "Vision Service timeout - el servicio tardó demasiado en responder",
                status_code=504,
            ) from e
        except httpx.ConnectError as e:
            raise VisionServiceError(
                "No se pudo conectar con Vision Service",
                status_code=503,
            ) from e
        except httpx.HTTPStatusError as e:
            raise VisionServiceError(
                f"Vision Service error: {e.response.text}",
                status_code=e.response.status_code,
            ) from e


async def health_check() -> dict[str, Any]:
    """Check Vision Service health."""
    return await _make_request("GET", "/health")


async def predict_static(landmarks: list[list[float]]) -> dict[str, Any]:
    """
    Predict static letter from hand landmarks.

    Args:
        landmarks: 21 landmarks × 3 coordinates (x, y, z)

    Returns:
        PredictionResponse with letter, confidence, type
    """
    return await _make_request("POST", "/predict/static", {"landmarks": landmarks})


async def predict_dynamic(sequence: list[list[list[float]]]) -> dict[str, Any]:
    """
    Predict dynamic letter from sequence of frames.

    Args:
        sequence: 15 frames × 21 landmarks × 3 coordinates

    Returns:
        PredictionResponse with letter, confidence, type
    """
    return await _make_request("POST", "/predict/dynamic", {"sequence": sequence})


async def predict_words(sequence: list[list[list[float]]]) -> dict[str, Any]:
    """
    Predict word from sequence of frames.

    Args:
        sequence: 15 frames × 21 landmarks × 3 coordinates

    Returns:
        WordPredictionResponse with word, confidence, phrase, accepted
    """
    return await _make_request("POST", "/predict/words", {"sequence": sequence})


async def predict_holistic(landmarks: list[float]) -> dict[str, Any]:
    """
    Predict medical vocabulary word from holistic features.

    Args:
        landmarks: 226 holistic features (pose + both hands)

    Returns:
        WordPredictionResponse with word, confidence, phrase, accepted
    """
    return await _make_request("POST", "/predict/holistic", {"landmarks": landmarks})


async def clear_word_buffer() -> dict[str, Any]:
    """Clear word prediction buffer."""
    return await _make_request("POST", "/predict/words/clear")


async def clear_holistic_buffer() -> dict[str, Any]:
    """Clear holistic prediction buffer."""
    return await _make_request("POST", "/predict/holistic/clear")


async def get_word_buffer_stats() -> dict[str, Any]:
    """Get word buffer statistics."""
    return await _make_request("GET", "/predict/words/stats")
