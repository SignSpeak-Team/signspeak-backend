from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App info
    SERVICE_NAME: str = "API Gateway"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # Server
    API_GATEWAY_HOST: str = "0.0.0.0"
    API_GATEWAY_PORT: int = 8000

    # CORS
    CORS_ORIGINS: list[str] = ["*"]

    # Downstream services (Docker DNS)
    TRANSLATION_SERVICE_URL: str = "http://translation-service:8001"

    class Config:
        env_file = ".env"


settings = Settings()
