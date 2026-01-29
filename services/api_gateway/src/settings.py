import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App info
    SERVICE_NAME: str = "API Gateway"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Server (Railway usa PORT dinámico)
    API_GATEWAY_HOST: str = "0.0.0.0"
    API_GATEWAY_PORT: int = int(os.getenv("PORT", "8000"))

    # CORS
    CORS_ORIGINS: list[str] = ["*"]

    # Downstream services - Lee de env vars en producción
    TRANSLATION_SERVICE_URL: str = os.getenv(
        "TRANSLATION_SERVICE_URL", 
        "http://translation-service:8001"
    )
    VISION_SERVICE_URL: str = os.getenv(
        "VISION_SERVICE_URL",
        "http://vision-service:8002"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
