"""Fixtures compartidos para todos los tests."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Agregar Vision Service src al PYTHONPATH
VISION_SERVICE_PATH = (
    Path(__file__).parent.parent / "services" / "vision_service" / "src"
)
if str(VISION_SERVICE_PATH) not in sys.path:
    sys.path.insert(0, str(VISION_SERVICE_PATH))


@pytest.fixture
def mock_landmarks():
    """Landmarks de ejemplo para testing."""
    return [[0.5, 0.5, 0.0] for _ in range(21)]


# @pytest.fixture
# def auth_client():
#     """Cliente HTTP para Auth Service."""
#     # Auth service aún no existe
#     from services.auth_service.src.main import app
#     return TestClient(app)


@pytest.fixture
def vision_client():
    """Cliente HTTP para Vision Service."""
    from api.main import app

    return TestClient(app)
