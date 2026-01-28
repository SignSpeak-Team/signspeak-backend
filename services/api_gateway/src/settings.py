from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App info
    SERVICE_NAME: str = "API Gateway"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # Server
    API_GATEWAY_HOST: str = "0.0.0.0"
    API_GATEWAY_PORT: int = 8000

    # CORS - Configure for your React app domain in production
    CORS_ORIGINS: list[str] = ["*"]

    # Downstream services (Docker DNS)
    TRANSLATION_SERVICE_URL: str = "http://translation-service:8001"
    VISION_SERVICE_URL: str = "http://vision-service:8002"

    # HTTP Client Configuration
    HTTP_TIMEOUT: float = 30.0  # seconds
    HTTP_CONNECT_TIMEOUT: float = 5.0  # seconds

    class Config:
        env_file = ".env"


settings = Settings()
