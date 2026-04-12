import pytest
from fastapi.testclient import TestClient
import sys
import os

@pytest.fixture
def translation_client():
    # Remove any existing cached 'src' modules from other test collections
    for key in list(sys.modules.keys()):
        if key.startswith("src.") or key == "src":
            del sys.modules[key]
            
    # Add the translation service path precisely
    target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../services/translation_service"))
    sys.path.insert(0, target_path)
    
    from src.main import app
    client = TestClient(app)
    yield client
    
    # Cleanup to not pollute other tests
    if target_path in sys.path:
        sys.path.remove(target_path)
    for key in list(sys.modules.keys()):
        if key.startswith("src.") or key == "src":
            del sys.modules[key]

class TestTranslationEndpoint:
    """Tests para /api/v1/translate/."""

    def test_translate_text_success(self, translation_client):
        """Traducción de texto exitosa."""
        response = translation_client.post(
            "/api/v1/translate/",
            json={"text": "hola mundo"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "Traducción de: hola mundo" in data["text"]
        assert data["confidence"] == 0.93

    def test_translate_video_url_success(self, translation_client):
        """Traducción de video_url exitosa."""
        response = translation_client.post(
            "/api/v1/translate/",
            json={"video_url": "http://example.com/video.mp4"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "[ML] Hola, ¿cómo estás?"
        assert data["confidence"] == 0.95

    def test_translate_missing_info(self, translation_client):
        """Traducción falla sin texto ni video."""
        response = translation_client.post(
            "/api/v1/translate/",
            json={"text": "", "video_url": ""} # Empty fields
        )
        assert response.status_code == 422
        assert "Debes enviar text o video_url" in response.json()["detail"]

    def test_translate_empty_body(self, translation_client):
        """Traducción falla si el body no tiene la estructura requerida."""
        response = translation_client.post(
            "/api/v1/translate/",
            json={} # Empty json
        )
        # Assuming Pydantic catches missing fields or TranslateRequest allows optional fields
        # If both are optional, our custom 422 triggers
        # Let's just assert it is 422
        assert response.status_code == 422
