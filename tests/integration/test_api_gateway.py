"""Integration tests para API Gateway → Vision Service.

Estos tests verifican que el API Gateway se comunica
correctamente con el Vision Service como proxy.
"""

import pytest


class TestAPIGatewayHealth:
    """Tests de health del API Gateway."""

    def test_gateway_health(self, gateway_sync_client):
        """API Gateway health check responde correctamente."""
        response = gateway_sync_client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_gateway_root(self, gateway_sync_client):
        """API Gateway root responde con info del servicio."""
        response = gateway_sync_client.get("/api/v1/")
        
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "endpoints" in data

    def test_gateway_status_shows_vision_healthy(self, gateway_sync_client):
        """API Gateway status muestra Vision Service como healthy."""
        response = gateway_sync_client.get("/api/v1/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "active_services" in data
        # Vision Service debería estar healthy si está corriendo
        assert data["active_services"]["vision_service"] in ["healthy", "unhealthy"]


class TestAPIGatewayToVisionProxy:
    """Tests de proxy de API Gateway a Vision Service."""

    def test_static_prediction_through_gateway(self, gateway_sync_client, sample_landmarks):
        """Predicción estática a través del API Gateway."""
        response = gateway_sync_client.post(
            "/api/v1/predict/static",
            json={"landmarks": sample_landmarks}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "letter" in data
        assert "confidence" in data

    def test_dynamic_prediction_through_gateway(self, gateway_sync_client, sample_sequence):
        """Predicción dinámica a través del API Gateway."""
        response = gateway_sync_client.post(
            "/api/v1/predict/dynamic",
            json={"sequence": sample_sequence}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "letter" in data

    def test_words_prediction_through_gateway(self, gateway_sync_client, sample_sequence):
        """Predicción de palabras a través del API Gateway."""
        response = gateway_sync_client.post(
            "/api/v1/predict/words",
            json={"sequence": sample_sequence}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "word" in data
        assert "phrase" in data

    def test_holistic_prediction_through_gateway(self, gateway_sync_client, sample_holistic_features):
        """Predicción holística a través del API Gateway."""
        response = gateway_sync_client.post(
            "/api/v1/predict/holistic",
            json={"landmarks": sample_holistic_features}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "word" in data


class TestAPIGatewayBufferManagement:
    """Tests de gestión de buffers a través del Gateway."""

    def test_get_word_stats_through_gateway(self, gateway_sync_client):
        """Obtener estadísticas del buffer a través del Gateway."""
        response = gateway_sync_client.get("/api/v1/predict/words/stats")
        
        assert response.status_code == 200

    def test_clear_word_buffer_through_gateway(self, gateway_sync_client):
        """Limpiar buffer de palabras a través del Gateway."""
        response = gateway_sync_client.post("/api/v1/predict/words/clear")
        
        assert response.status_code == 200

    def test_clear_holistic_buffer_through_gateway(self, gateway_sync_client):
        """Limpiar buffer holístico a través del Gateway."""
        response = gateway_sync_client.post("/api/v1/predict/holistic/clear")
        
        assert response.status_code == 200


class TestAPIGatewayErrorHandling:
    """Tests de manejo de errores del API Gateway."""

    def test_invalid_landmarks_returns_422(self, gateway_sync_client):
        """Error 422 con landmarks inválidos."""
        response = gateway_sync_client.post(
            "/api/v1/predict/static",
            json={"landmarks": [[0.5]]}  # Inválido
        )
        
        assert response.status_code == 422

    def test_invalid_sequence_returns_422(self, gateway_sync_client):
        """Error 422 con secuencia inválida."""
        response = gateway_sync_client.post(
            "/api/v1/predict/dynamic",
            json={"sequence": []}  # Inválido
        )
        
        assert response.status_code == 422

    def test_invalid_holistic_features_returns_422(self, gateway_sync_client):
        """Error 422 con features holísticas inválidas."""
        response = gateway_sync_client.post(
            "/api/v1/predict/holistic",
            json={"landmarks": [0.5] * 100}  # Solo 100, necesita 226
        )
        
        assert response.status_code == 422
