"""Fixtures para Integration Tests.

Carga configuración desde .env y proporciona clientes HTTP
para comunicación con servicios reales.
"""

import os
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

# Cargar .env desde directorio de integration tests
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

# URLs de servicios (configurable via .env)
VISION_SERVICE_URL = os.getenv("VISION_SERVICE_URL", "http://localhost:8002")
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def vision_base_url():
    """URL base del Vision Service."""
    return VISION_SERVICE_URL


@pytest.fixture(scope="session")
def gateway_base_url():
    """URL base del API Gateway."""
    return API_GATEWAY_URL


@pytest.fixture
def vision_http_client(vision_base_url):
    """Cliente HTTP para Vision Service (async)."""
    return httpx.AsyncClient(base_url=vision_base_url, timeout=30.0)


@pytest.fixture
def gateway_http_client(gateway_base_url):
    """Cliente HTTP para API Gateway (async)."""
    return httpx.AsyncClient(base_url=gateway_base_url, timeout=30.0)


@pytest.fixture
def vision_sync_client(vision_base_url):
    """Cliente HTTP síncrono para Vision Service."""
    return httpx.Client(base_url=vision_base_url, timeout=30.0)


@pytest.fixture
def gateway_sync_client(gateway_base_url):
    """Cliente HTTP síncrono para API Gateway."""
    return httpx.Client(base_url=gateway_base_url, timeout=30.0)


# === Test Data Fixtures ===


@pytest.fixture
def sample_landmarks():
    """21 landmarks x 3 coordenadas para testing."""
    return [[0.5, 0.5, 0.0] for _ in range(21)]


@pytest.fixture
def sample_sequence():
    """15 frames x 21 landmarks x 3 coordenadas."""
    landmarks = [[0.5, 0.5, 0.0] for _ in range(21)]
    return [landmarks for _ in range(15)]


@pytest.fixture
def sample_holistic_features():
    """226 features holísticas para testing."""
    return [0.5] * 226


# === Service Health Check ===


@pytest.fixture(scope="session", autouse=True)
def check_services_availability(vision_base_url, gateway_base_url):
    """Verifica que los servicios estén disponibles antes de correr tests."""
    services = {
        "Vision Service": vision_base_url,
        "API Gateway": gateway_base_url,
    }

    unavailable = []

    for name, url in services.items():
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{url}/api/v1/health")
                if response.status_code != 200:
                    unavailable.append(f"{name} ({url}): status {response.status_code}")
        except httpx.ConnectError:
            unavailable.append(f"{name} ({url}): no conecta")
        except Exception as e:
            unavailable.append(f"{name} ({url}): {e}")

    if unavailable:
        pytest.skip(
            "Servicios no disponibles:\n"
            + "\n".join(unavailable)
            + "\n\nAsegúrate de que los servicios estén corriendo."
        )
