"""Tests unitarios para Vision Service."""
import pytest


def test_health_endpoint(vision_client):
    """Test que el endpoint de health funciona."""
    response = vision_client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_static_success(vision_client, mock_landmarks):
    """Test predicción de letra estática."""
    payload = {
        "landmarks": mock_landmarks  # 21 landmarks x 3 coords
    }
    
    response = vision_client.post(
        "/api/v1/predict/static",
        json=payload
    )
    
    assert response.status_code == 200
    data = response.json()
    # Verificar estructura de respuesta real
    assert "letter" in data
    assert "confidence" in data
    assert "type" in data
    assert data["type"] == "static"
    assert 0.0 <= data["confidence"] <= 100.0


def test_predict_static_invalid_landmarks(vision_client):
    """Test con landmarks inválidos."""
    payload = {
        "landmarks": [[1, 2]]  # Solo 1 landmark, se necesitan 21
    }
    
    response = vision_client.post(
        "/api/v1/predict/static",
        json=payload
    )
    
    # Pydantic valida min_length=21
    assert response.status_code == 422  # Validation error


@pytest.fixture
def mock_sequence():
    """Genera una secuencia de 15 frames x 21 landmarks para testing."""
    landmarks = [[0.5, 0.5, 0.0] for _ in range(21)]
    return [landmarks for _ in range(15)]  # 15 frames


@pytest.mark.slow
def test_predict_words_full_sequence(vision_client, mock_sequence):
    """Test predicción de palabras con secuencia completa."""
    payload = {"sequence": mock_sequence}  # 15 frames x 21 landmarks x 3 coords
    
    response = vision_client.post(
        "/api/v1/predict/words",
        json=payload
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "word" in data
    assert "confidence" in data
    assert "phrase" in data