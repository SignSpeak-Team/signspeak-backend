import os
from pathlib import Path

from pydantic_settings import BaseSettings

# Raíz del proyecto (3 niveles arriba de src/settings.py)
ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # App info
    SERVICE_NAME: str = "API Gateway"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Server — HF Spaces uses port 7860 by default
    API_GATEWAY_HOST: str = "0.0.0.0"
    API_GATEWAY_PORT: int = int(os.getenv("PORT", "7860"))

    # CORS - Configure for your React app domain in production
    CORS_ORIGINS: list[str] = ["*"]

    # Downstream services - Lee de env vars en producción
    TRANSLATION_SERVICE_URL: str = os.getenv(
        "TRANSLATION_SERVICE_URL", "http://translation-service:8001"
    )
    VISION_SERVICE_URL: str = os.getenv(
        "VISION_SERVICE_URL", "http://vision-service:8001"
    )

    # HTTP Client Configuration
    HTTP_TIMEOUT: float = 30.0  # seconds
    HTTP_CONNECT_TIMEOUT: float = 5.0  # seconds

    class Config:
        # Carga el .env desde la raíz del proyecto
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = "utf-8"


settings = Settings()
