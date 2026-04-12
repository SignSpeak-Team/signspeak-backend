from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Service info
    SERVICE_NAME: str = "Translation Service"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # Server — Cloud Run usa $PORT dinámicamente
    PORT: int = 8080

    # Downstream services — se inyectan como env vars en Cloud Run
    VISION_SERVICE_URL: str = "http://vision-service:8080"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
