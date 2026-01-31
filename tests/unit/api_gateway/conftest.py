"""Fixtures específicos para API Gateway tests."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

# Agregar API Gateway src al PYTHONPATH (el padre de src)
API_GATEWAY_PATH = Path(__file__).parent.parent.parent.parent / "services" / "api_gateway"
if str(API_GATEWAY_PATH) not in sys.path:
    sys.path.insert(0, str(API_GATEWAY_PATH))


@pytest.fixture
def gateway_client():
    """Cliente HTTP para API Gateway."""
    from src.main import app
    return TestClient(app)


@pytest.fixture
def mock_landmarks():
    """21 landmarks x 3 coordenadas para testing."""
    return [[0.5, 0.5, 0.0] for _ in range(21)]


@pytest.fixture
def mock_sequence():
    """15 frames x 21 landmarks x 3 coordenadas."""
    landmarks = [[0.5, 0.5, 0.0] for _ in range(21)]
    return [landmarks for _ in range(15)]


@pytest.fixture
def mock_holistic_landmarks():
    """226 features holísticas."""
    return [0.5] * 226


@pytest.fixture
def mock_vision_response():
    """Respuesta mockeada de Vision Service para letters."""
    return {
        "letter": "A",
        "confidence": 95.5,
        "type": "static",
        "processing_time_ms": 10
    }


@pytest.fixture
def mock_word_response():
    """Respuesta mockeada de Vision Service para words."""
    return {
        "word": "hola",
        "confidence": 90.0,
        "phrase": "hola mundo",
        "accepted": True,
        "processing_time_ms": 15
    }


@pytest.fixture
def mock_vision_client():
    """Mock del vision_client para evitar llamadas HTTP reales."""
    with patch("src.routes.prediction.predict_static", new_callable=AsyncMock) as static, \
         patch("src.routes.prediction.predict_dynamic", new_callable=AsyncMock) as dynamic, \
         patch("src.routes.prediction.predict_words", new_callable=AsyncMock) as words, \
         patch("src.routes.prediction.predict_holistic", new_callable=AsyncMock) as holistic, \
         patch("src.routes.prediction.get_word_buffer_stats", new_callable=AsyncMock) as stats, \
         patch("src.routes.prediction.clear_word_buffer", new_callable=AsyncMock) as clear_word, \
         patch("src.routes.prediction.clear_holistic_buffer", new_callable=AsyncMock) as clear_holistic, \
         patch("src.services.vision_client.health_check", new_callable=AsyncMock) as health:
        
        yield {
            "predict_static": static,
            "predict_dynamic": dynamic,
            "predict_words": words,
            "predict_holistic": holistic,
            "get_word_buffer_stats": stats,
            "clear_word_buffer": clear_word,
            "clear_holistic_buffer": clear_holistic,
            "health_check": health,
        }
