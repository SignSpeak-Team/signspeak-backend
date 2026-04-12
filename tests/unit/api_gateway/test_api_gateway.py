"""Tests unitarios para API Gateway endpoints.

Estos tests mockean las llamadas al Vision Service para probar
la lógica del API Gateway de forma aislada.
"""

import pytest


class TestHealthEndpoints:
    """Tests para endpoints de health."""

    def test_root_returns_service_info(self, gateway_client):
        """Root retorna información del servicio."""
        response = gateway_client.get("/api/v1/")

        assert response.status_code == 200
        data = response.json()
        assert "API Gateway" in data["service"]  # Matches settings.SERVICE_NAME
        assert "version" in data
        assert data["status"] == "running"
        assert "endpoints" in data

    def test_health_returns_healthy(self, gateway_client):
        """Health check retorna estado saludable."""
        response = gateway_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_status_returns_system_info(self, gateway_client, mock_vision_client):
        """Status retorna información del sistema."""
        mock_vision_client["health_check"].return_value = {"status": "healthy"}

        response = gateway_client.get("/api/v1/status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime" in data
        assert "active_services" in data

    def test_status_degraded_when_vision_unhealthy(
        self, gateway_client, mock_vision_client
    ):
        """Status degraded cuando Vision Service no responde."""
        mock_vision_client["health_check"].side_effect = Exception("Connection error")

        response = gateway_client.get("/api/v1/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["active_services"]["vision_service"] == "unhealthy"


class TestStaticPredictionEndpoint:
    """Tests para /predict/static."""

    def test_predict_static_success(
        self, gateway_client, mock_vision_client, mock_landmarks, mock_vision_response
    ):
        """Predicción estática exitosa."""
        mock_vision_client["predict_static"].return_value = mock_vision_response

        response = gateway_client.post(
            "/api/v1/predict/static", json={"landmarks": mock_landmarks}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["letter"] == "A"
        assert data["confidence"] == 95.5

    def test_predict_static_invalid_landmarks(self, gateway_client):
        """Error con landmarks inválidos."""
        response = gateway_client.post(
            "/api/v1/predict/static", json={"landmarks": [[0.5, 0.5]]}  # Solo 2 coords
        )

        assert response.status_code == 422


class TestDynamicPredictionEndpoint:
    """Tests para /predict/dynamic."""

    def test_predict_dynamic_success(
        self, gateway_client, mock_vision_client, mock_sequence, mock_vision_response
    ):
        """Predicción dinámica exitosa."""
        mock_vision_response["type"] = "dynamic"
        mock_vision_client["predict_dynamic"].return_value = mock_vision_response

        response = gateway_client.post(
            "/api/v1/predict/dynamic", json={"sequence": mock_sequence}
        )

        assert response.status_code == 200
        data = response.json()
        assert "letter" in data
        assert "confidence" in data

    def test_predict_dynamic_invalid_sequence(self, gateway_client):
        """Error con secuencia inválida."""
        response = gateway_client.post("/api/v1/predict/dynamic", json={"sequence": []})

        assert response.status_code == 422


class TestWordPredictionEndpoint:
    """Tests para /predict/words."""

    def test_predict_word_success(
        self, gateway_client, mock_vision_client, mock_sequence, mock_word_response
    ):
        """Predicción de palabra exitosa."""
        mock_vision_client["predict_words"].return_value = mock_word_response

        response = gateway_client.post(
            "/api/v1/predict/words", json={"sequence": mock_sequence}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["word"] == "hola"
        assert "phrase" in data


class TestHolisticPredictionEndpoint:
    """Tests para /predict/holistic."""

    def test_predict_holistic_success(
        self,
        gateway_client,
        mock_vision_client,
        mock_holistic_landmarks,
        mock_word_response,
    ):
        """Predicción holística exitosa."""
        mock_vision_client["predict_holistic"].return_value = mock_word_response

        response = gateway_client.post(
            "/api/v1/predict/holistic", json={"landmarks": mock_holistic_landmarks}
        )

        assert response.status_code == 200
        data = response.json()
        assert "word" in data

    def test_predict_holistic_invalid_features(self, gateway_client):
        """Error con número incorrecto de features."""
        response = gateway_client.post(
            "/api/v1/predict/holistic",
            json={"landmarks": [0.5] * 100},  # Solo 100, necesita 226
        )

        assert response.status_code == 422


class TestBufferManagementEndpoints:
    """Tests para endpoints de gestión de buffers."""

    def test_get_buffer_stats(self, gateway_client, mock_vision_client):
        """Obtener estadísticas del buffer."""
        mock_vision_client["get_word_buffer_stats"].return_value = {
            "total_received": 15,
            "total_accepted": 3,
            "rejected_by_cooldown": 5,
            "rejected_by_confidence": 7,
            "acceptance_rate": 20.0,
            "current_phrase": "hola mundo adios",
        }

        response = gateway_client.get("/api/v1/predict/words/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_received"] == 15
        assert data["total_accepted"] == 3

    def test_clear_word_buffer(self, gateway_client, mock_vision_client):
        """Limpiar buffer de palabras."""
        mock_vision_client["clear_word_buffer"].return_value = {
            "message": "Buffer cleared"
        }

        response = gateway_client.post("/api/v1/predict/words/clear")

        assert response.status_code == 200

    def test_clear_holistic_buffer(self, gateway_client, mock_vision_client):
        """Limpiar buffer holístico."""
        mock_vision_client["clear_holistic_buffer"].return_value = {
            "message": "Buffer cleared"
        }

        response = gateway_client.post("/api/v1/predict/holistic/clear")

        assert response.status_code == 200


class TestErrorHandling:
    """Tests para manejo de errores del Vision Service."""

    def test_vision_service_timeout(
        self, gateway_client, mock_vision_client, mock_landmarks
    ):
        """Manejo de timeout del Vision Service."""
        from src.services.vision_client import VisionServiceError

        mock_vision_client["predict_static"].side_effect = VisionServiceError(
            "Timeout", status_code=504
        )

        response = gateway_client.post(
            "/api/v1/predict/static", json={"landmarks": mock_landmarks}
        )

        assert response.status_code == 504

    def test_vision_service_unavailable(
        self, gateway_client, mock_vision_client, mock_landmarks
    ):
        """Manejo de Vision Service no disponible."""
        from src.services.vision_client import VisionServiceError

        mock_vision_client["predict_static"].side_effect = VisionServiceError(
            "Connection refused", status_code=503
        )

        response = gateway_client.post(
            "/api/v1/predict/static", json={"landmarks": mock_landmarks}
        )

        assert response.status_code == 503
