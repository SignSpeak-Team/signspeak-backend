"""Fixtures específicos para Vision Service tests.

NOTE: Los tests que usen `vision_client` cargan TF/Keras/MediaPipe.
      Están marcados como @pytest.mark.slow y se excluyen del CI ligero.
      Para correrlos localmente: pytest -m slow
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Agregar Vision Service src al PYTHONPATH
VISION_SERVICE_PATH = Path(__file__).parent.parent.parent.parent / "services" / "vision_service" / "src"
if str(VISION_SERVICE_PATH) not in sys.path:
    sys.path.insert(0, str(VISION_SERVICE_PATH))


@pytest.fixture
@pytest.mark.slow
def vision_client():
    """Cliente HTTP para Vision Service."""
    from api.main import app
    return TestClient(app)


@pytest.fixture
def mock_landmarks():
    """Landmarks de ejemplo para testing (21 x 3)."""
    return [[0.5, 0.5, 0.0] for _ in range(21)]


@pytest.fixture
def mock_sequence():
    """Secuencia de 15 frames x 21 landmarks para testing."""
    landmarks = [[0.5, 0.5, 0.0] for _ in range(21)]
    return [landmarks for _ in range(15)]


@pytest.fixture
def mock_holistic_landmarks():
    """226 features para predicción holística."""
    return [0.5] * 226
