"""Tests para SignPredictor via API — todos marcados slow (requieren TF/Keras/MediaPipe)."""

import sys
from pathlib import Path

import numpy as np
import pytest

VISION_SERVICE_PATH = (
    Path(__file__).parent.parent.parent.parent / "services" / "vision_service" / "src"
)
if str(VISION_SERVICE_PATH) not in sys.path:
    sys.path.insert(0, str(VISION_SERVICE_PATH))


@pytest.mark.slow
class TestStaticPredictionAPI:

    def test_predict_static_valid_landmarks(self, vision_client, mock_landmarks):
        response = vision_client.post(
            "/api/v1/predict/static", json={"landmarks": mock_landmarks}
        )

        assert response.status_code == 200
        data = response.json()
        assert "letter" in data
        assert "confidence" in data
        assert data["type"] == "static"
        assert 0 <= data["confidence"] <= 100

    def test_predict_static_wrong_landmark_count(self, vision_client):
        landmarks = [[0.5, 0.5, 0.0] for _ in range(10)]
        response = vision_client.post(
            "/api/v1/predict/static", json={"landmarks": landmarks}
        )
        assert response.status_code == 422

    def test_predict_static_wrong_coord_count(self, vision_client):
        landmarks = [[0.5, 0.5] for _ in range(21)]
        response = vision_client.post(
            "/api/v1/predict/static", json={"landmarks": landmarks}
        )
        assert response.status_code == 422

    def test_predict_static_empty_request(self, vision_client):
        response = vision_client.post("/api/v1/predict/static", json={})
        assert response.status_code == 422


@pytest.mark.slow
class TestDynamicPredictionAPI:

    def test_predict_dynamic_valid_sequence(self, vision_client, mock_sequence):
        response = vision_client.post(
            "/api/v1/predict/dynamic", json={"sequence": mock_sequence}
        )

        assert response.status_code == 200
        data = response.json()
        assert "letter" in data
        assert "confidence" in data
        assert data["type"] == "dynamic"

    def test_predict_dynamic_wrong_frame_count(self, vision_client):
        landmarks = [[0.5, 0.5, 0.0] for _ in range(21)]
        sequence = [landmarks for _ in range(10)]
        response = vision_client.post(
            "/api/v1/predict/dynamic", json={"sequence": sequence}
        )
        assert response.status_code == 422

    def test_predict_dynamic_wrong_landmarks_per_frame(self, vision_client):
        landmarks = [[0.5, 0.5, 0.0] for _ in range(10)]
        sequence = [landmarks for _ in range(15)]
        response = vision_client.post(
            "/api/v1/predict/dynamic", json={"sequence": sequence}
        )
        assert response.status_code == 422


@pytest.mark.slow
class TestWordsPredictionAPI:

    def test_predict_words_valid_sequence(self, vision_client, mock_sequence):
        response = vision_client.post(
            "/api/v1/predict/words", json={"sequence": mock_sequence}
        )

        assert response.status_code == 200
        data = response.json()
        assert "word" in data
        assert "confidence" in data
        assert "phrase" in data

    def test_predict_words_invalid_sequence(self, vision_client):
        response = vision_client.post("/api/v1/predict/words", json={"sequence": []})
        assert response.status_code == 422


@pytest.mark.slow
class TestHolisticPredictionAPI:

    @pytest.fixture
    def mock_holistic_landmarks(self):
        return [0.5] * 226

    def test_predict_holistic_valid_landmarks(
        self, vision_client, mock_holistic_landmarks
    ):
        response = vision_client.post(
            "/api/v1/predict/holistic", json={"landmarks": mock_holistic_landmarks}
        )

        assert response.status_code == 200
        data = response.json()
        assert "word" in data
        assert "confidence" in data

    def test_predict_holistic_wrong_feature_count(self, vision_client):
        response = vision_client.post(
            "/api/v1/predict/holistic", json={"landmarks": [0.5] * 100}
        )
        assert response.status_code == 422


@pytest.mark.slow
class TestBufferManagementAPI:

    def test_get_word_buffer_stats(self, vision_client):
        response = vision_client.get("/api/v1/predict/words/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_received" in data
        assert "total_accepted" in data
        assert "acceptance_rate" in data

    def test_clear_word_buffer(self, vision_client):
        response = vision_client.post("/api/v1/predict/words/clear")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_clear_holistic_buffer(self, vision_client):
        response = vision_client.post("/api/v1/predict/holistic/clear")
        assert response.status_code == 200
        assert "message" in response.json()


@pytest.mark.slow
class TestHealthEndpoint:

    def test_health_returns_healthy(self, vision_client):
        response = vision_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "version" in data

    def test_health_models_loaded(self, vision_client):
        response = vision_client.get("/api/v1/health")
        assert response.json()["models_loaded"] is True


@pytest.mark.slow
class TestRootEndpoint:

    def test_root_returns_info(self, vision_client):
        response = vision_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "SignSpeak Vision API"
        assert "version" in data
        assert data["status"] == "running"


@pytest.fixture
def mock_sequence():
    landmarks = [[0.5, 0.5, 0.0] for _ in range(21)]
    return [landmarks for _ in range(15)]
