"""Integration tests for Media Processing Pipeline."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add vision_service/src to path
VISION_SRC = Path(__file__).parents[2] / "services" / "vision_service" / "src"
sys.path.append(str(VISION_SRC))

import numpy as np
import pytest
from api.main import app
from api.routes.media import get_video_processor
from fastapi.testclient import TestClient

# Create mock processor
mock_processor = MagicMock()
mock_processor.process_video_bytes.return_value = np.zeros((30, 226), dtype=np.float32)


def get_mock_video_processor():
    """Dependency override."""
    return mock_processor


@pytest.fixture
def client_with_mock():
    """Test client with mocked video processor."""
    app.dependency_overrides[get_video_processor] = get_mock_video_processor
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_translate_video_endpoint(client_with_mock):
    """Test /media/translate/video endpoint with mocked processor."""
    # Mock for continuous mode which is the default
    mock_processor.process_video_sliding_window.return_value = [
        {"start_time": 0, "end_time": 1, "features": np.zeros(226)}
    ]
    
    video_content = b"fake-video-content"
    response = client_with_mock.post(
        "/api/v1/media/translate/video",
        data={"mode": "continuous"},
        files={"file": ("test.mp4", video_content, "video/mp4")},
    )

    assert response.status_code == 200
    data = response.json()

    assert "word" in data
    assert "confidence" in data
    assert "extraction_time_ms" in data
    assert "prediction_time_ms" in data
    assert "total_time_ms" in data
    assert data["frames_processed"] == 30

    # Verify mock was called
    mock_processor.process_video_sliding_window.assert_called_once()


def test_translate_video_invalid_file_type(client_with_mock):
    """Test invalid file type rejection."""
    response = client_with_mock.post(
        "/api/v1/media/translate/video",
        files={"file": ("test.txt", b"text content", "text/plain")},
    )

    assert response.status_code == 400
    assert "Invalid file type" in response.text
