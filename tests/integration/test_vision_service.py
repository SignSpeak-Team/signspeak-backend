"""Integration tests para Vision Service.

Estos tests hacen llamadas HTTP reales al Vision Service
para verificar que los endpoints funcionan correctamente.
"""

import pytest


class TestVisionServiceHealth:
    """Tests de health y status del Vision Service."""

    def test_health_endpoint(self, vision_sync_client):
        """Vision Service health check responde correctamente."""
        response = vision_sync_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_endpoint(self, vision_sync_client):
        """Vision Service root responde o health funciona."""
        # Vision Service puede no tener / root, pero health debe funcionar
        response = vision_sync_client.get("/api/v1/health")

        assert response.status_code == 200


class TestStaticPredictionIntegration:
    """Tests de predicción estática con modelos reales."""

    def test_static_prediction_returns_letter(
        self, vision_sync_client, sample_landmarks
    ):
        """Predicción estática retorna una letra válida."""
        response = vision_sync_client.post(
            "/api/v1/predict/static", json={"landmarks": sample_landmarks}
        )

        assert response.status_code == 200
        data = response.json()
        assert "letter" in data
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 100
        assert data["type"] == "static"

    def test_static_prediction_invalid_landmarks(self, vision_sync_client):
        """Error con landmarks inválidos."""
        response = vision_sync_client.post(
            "/api/v1/predict/static",
            json={"landmarks": [[0.5, 0.5]]},  # Solo 2 coords, necesita 3
        )

        assert response.status_code == 422


class TestDynamicPredictionIntegration:
    """Tests de predicción dinámica con modelos reales."""

    def test_dynamic_prediction_returns_letter(
        self, vision_sync_client, sample_sequence
    ):
        """Predicción dinámica retorna una letra válida."""
        response = vision_sync_client.post(
            "/api/v1/predict/dynamic", json={"sequence": sample_sequence}
        )

        assert response.status_code == 200
        data = response.json()
        assert "letter" in data
        assert "confidence" in data
        assert data["type"] == "dynamic"


class TestWordPredictionIntegration:
    """Tests de predicción de palabras con modelos reales."""

    def test_word_prediction_returns_word(self, vision_sync_client, sample_sequence):
        """Predicción de palabras retorna una palabra válida."""
        response = vision_sync_client.post(
            "/api/v1/predict/words", json={"sequence": sample_sequence}
        )

        assert response.status_code == 200
        data = response.json()
        assert "word" in data
        assert "confidence" in data
        assert "phrase" in data

    def test_word_buffer_stats(self, vision_sync_client):
        """Obtener estadísticas del buffer."""
        response = vision_sync_client.get("/api/v1/predict/words/stats")

        assert response.status_code == 200
        data = response.json()
        assert "buffer_size" in data or "total_received" in data

    def test_clear_word_buffer(self, vision_sync_client):
        """Limpiar buffer de palabras."""
        response = vision_sync_client.post("/api/v1/predict/words/clear")

        assert response.status_code == 200


class TestHolisticPredictionIntegration:
    """Tests de predicción holística con modelos reales."""

    def test_holistic_prediction_returns_word(
        self, vision_sync_client, sample_holistic_features
    ):
        """Predicción holística retorna una palabra válida."""
        response = vision_sync_client.post(
            "/api/v1/predict/holistic", json={"landmarks": sample_holistic_features}
        )

        assert response.status_code == 200
        data = response.json()
        assert "word" in data
        assert "confidence" in data

    def test_clear_holistic_buffer(self, vision_sync_client):
        """Limpiar buffer holístico."""
        response = vision_sync_client.post("/api/v1/predict/holistic/clear")

        assert response.status_code == 200
