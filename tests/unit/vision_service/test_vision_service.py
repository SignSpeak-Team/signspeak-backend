"""Tests unitarios para Vision Service API — solo validaciones, sin cargar modelos."""

import pytest


@pytest.mark.slow
def test_health_endpoint(vision_client):
    response = vision_client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.slow
def test_predict_static_success(vision_client, mock_landmarks):
    response = vision_client.post(
        "/api/v1/predict/static", json={"landmarks": mock_landmarks}
    )

    assert response.status_code == 200
    data = response.json()
    assert "letter" in data
    assert "confidence" in data
    assert data["type"] == "static"
    assert 0.0 <= data["confidence"] <= 100.0


@pytest.mark.slow
def test_predict_static_invalid_landmarks(vision_client):
    response = vision_client.post(
        "/api/v1/predict/static", json={"landmarks": [[1, 2]]}
    )
    assert response.status_code == 422


@pytest.fixture
def mock_sequence():
    landmarks = [[0.5, 0.5, 0.0] for _ in range(21)]
    return [landmarks for _ in range(15)]


@pytest.mark.slow
def test_predict_words_full_sequence(vision_client, mock_sequence):
    response = vision_client.post(
        "/api/v1/predict/words", json={"sequence": mock_sequence}
    )

    assert response.status_code == 200
    data = response.json()
    assert "word" in data
    assert "confidence" in data
    assert "phrase" in data
