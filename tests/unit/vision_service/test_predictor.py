"""Tests unitarios para SignPredictor - Core del Vision Service.

NOTA: Estos tests usan el predictor singleton que ya está cargado
en el conftest.py via vision_client, evitando problemas de carga
de modelos múltiples veces.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Agregar vision_service al path
VISION_SERVICE_PATH = Path(__file__).parent.parent.parent.parent / "services" / "vision_service" / "src"
if str(VISION_SERVICE_PATH) not in sys.path:
    sys.path.insert(0, str(VISION_SERVICE_PATH))


# ============================================================================
# TESTS VIA API (más seguros - no cargan modelos directamente)
# ============================================================================

class TestStaticPredictionAPI:
    """Tests para predicción estática via API."""

    def test_predict_static_valid_landmarks(self, vision_client, mock_landmarks):
        """Predicción con landmarks válidos."""
        response = vision_client.post(
            "/api/v1/predict/static",
            json={"landmarks": mock_landmarks}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "letter" in data
        assert "confidence" in data
        assert "type" in data
        assert data["type"] == "static"
        assert 0 <= data["confidence"] <= 100

    def test_predict_static_wrong_landmark_count(self, vision_client):
        """Error con número incorrecto de landmarks."""
        landmarks = [[0.5, 0.5, 0.0] for _ in range(10)]  # Solo 10, necesita 21
        
        response = vision_client.post(
            "/api/v1/predict/static",
            json={"landmarks": landmarks}
        )
        
        assert response.status_code == 422

    def test_predict_static_wrong_coord_count(self, vision_client):
        """Error con número incorrecto de coordenadas."""
        landmarks = [[0.5, 0.5] for _ in range(21)]  # Solo 2 coords, necesita 3
        
        response = vision_client.post(
            "/api/v1/predict/static",
            json={"landmarks": landmarks}
        )
        
        assert response.status_code == 422

    def test_predict_static_empty_request(self, vision_client):
        """Error con request vacío."""
        response = vision_client.post(
            "/api/v1/predict/static",
            json={}
        )
        
        assert response.status_code == 422


class TestDynamicPredictionAPI:
    """Tests para predicción dinámica via API."""

    def test_predict_dynamic_valid_sequence(self, vision_client, mock_sequence):
        """Predicción dinámica con secuencia válida."""
        response = vision_client.post(
            "/api/v1/predict/dynamic",
            json={"sequence": mock_sequence}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "letter" in data
        assert "confidence" in data
        assert data["type"] == "dynamic"

    def test_predict_dynamic_wrong_frame_count(self, vision_client):
        """Error con número incorrecto de frames."""
        landmarks = [[0.5, 0.5, 0.0] for _ in range(21)]
        sequence = [landmarks for _ in range(10)]  # Solo 10, necesita 15
        
        response = vision_client.post(
            "/api/v1/predict/dynamic",
            json={"sequence": sequence}
        )
        
        assert response.status_code == 422

    def test_predict_dynamic_wrong_landmarks_per_frame(self, vision_client):
        """Error con landmarks incorrectos por frame."""
        landmarks = [[0.5, 0.5, 0.0] for _ in range(10)]  # Solo 10
        sequence = [landmarks for _ in range(15)]
        
        response = vision_client.post(
            "/api/v1/predict/dynamic",
            json={"sequence": sequence}
        )
        
        assert response.status_code == 422


class TestWordsPredictionAPI:
    """Tests para predicción de palabras via API."""

    @pytest.mark.slow
    def test_predict_words_valid_sequence(self, vision_client, mock_sequence):
        """Predicción de palabras con secuencia válida."""
        response = vision_client.post(
            "/api/v1/predict/words",
            json={"sequence": mock_sequence}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "word" in data
        assert "confidence" in data
        assert "phrase" in data

    def test_predict_words_invalid_sequence(self, vision_client):
        """Error con secuencia inválida."""
        response = vision_client.post(
            "/api/v1/predict/words",
            json={"sequence": []}
        )
        
        assert response.status_code == 422


class TestHolisticPredictionAPI:
    """Tests para predicción holística via API."""

    @pytest.fixture
    def mock_holistic_landmarks(self):
        """226 features para predicción holística."""
        return [0.5] * 226

    def test_predict_holistic_valid_landmarks(self, vision_client, mock_holistic_landmarks):
        """Predicción holística con landmarks válidos."""
        response = vision_client.post(
            "/api/v1/predict/holistic",
            json={"landmarks": mock_holistic_landmarks}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "word" in data
        assert "confidence" in data

    def test_predict_holistic_wrong_feature_count(self, vision_client):
        """Error con número incorrecto de features."""
        landmarks = [0.5] * 100  # Solo 100, necesita 226
        
        response = vision_client.post(
            "/api/v1/predict/holistic",
            json={"landmarks": landmarks}
        )
        
        assert response.status_code == 422


class TestBufferManagementAPI:
    """Tests para gestión de buffers via API."""

    def test_get_word_buffer_stats(self, vision_client):
        """Obtener estadísticas del buffer de palabras."""
        response = vision_client.get("/api/v1/predict/words/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_received" in data
        assert "total_accepted" in data
        assert "acceptance_rate" in data

    def test_clear_word_buffer(self, vision_client):
        """Limpiar buffer de palabras."""
        response = vision_client.post("/api/v1/predict/words/clear")
        
        assert response.status_code == 200
        assert "message" in response.json()

    def test_clear_holistic_buffer(self, vision_client):
        """Limpiar buffer holístico."""
        response = vision_client.post("/api/v1/predict/holistic/clear")
        
        assert response.status_code == 200
        assert "message" in response.json()


class TestHealthEndpoint:
    """Tests para endpoint de health."""

    def test_health_returns_healthy(self, vision_client):
        """Health check retorna estado saludable."""
        response = vision_client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "version" in data

    def test_health_models_loaded(self, vision_client):
        """Health indica que modelos están cargados."""
        response = vision_client.get("/api/v1/health")
        
        data = response.json()
        assert data["models_loaded"] is True


class TestRootEndpoint:
    """Tests para endpoint raíz."""

    def test_root_returns_info(self, vision_client):
        """Root retorna información del servicio."""
        response = vision_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "SignSpeak Vision API"
        assert "version" in data
        assert data["status"] == "running"


# ============================================================================
# FIXTURES ADICIONALES
# ============================================================================

@pytest.fixture
def mock_sequence():
    """Genera secuencia de 15 frames x 21 landmarks para testing."""
    landmarks = [[0.5, 0.5, 0.0] for _ in range(21)]
    return [landmarks for _ in range(15)]
