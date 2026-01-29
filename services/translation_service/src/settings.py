import os
from pydantic import BaseModel


class Settings(BaseModel):
    # Service info
    SERVICE_NAME: str = "Translation Service"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Server
    PORT: int = int(os.getenv("PORT", "8001"))
    
    # Downstream services
    VISION_SERVICE_URL: str = os.getenv(
        "VISION_SERVICE_URL",
        "http://vision-service:8002"
    )


settings = Settings()
