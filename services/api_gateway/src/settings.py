from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Configuración centralizada del API Gateway

    Las variables se pueden definir en:
    1. Variables de entorno del sistema
    2. Archivo .env
    3. Valores por defecto (aquí)
    """

    # Información del servicio
    SERVICE_NAME: str = "API Gateway"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Configuración del servidor
    API_GATEWAY_HOST: str = "127.0.0.1"
    API_GATEWAY_PORT: int = 8000

    # URLs de otros microservicios
    TRANSLATION_SERVICE_URL: str = "http://localhost:8001"
    ML_SERVICE_URL: str = "http://localhost:8002"
    STORAGE_SERVICE_URL: str = "http://localhost:8003"

    # Autenticación JWT (lo configuraremos después)
    JWT_SECRET_KEY: str = "dev-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    # CORS (permitir requests desde el frontend)
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    class Config:
        """Configuración de Pydantic"""
        env_file = ".env"
        case_sensitive = True


settings = Settings()