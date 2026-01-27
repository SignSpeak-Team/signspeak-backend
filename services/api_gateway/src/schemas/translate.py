from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class TranslateRequest(BaseModel):
    video_url: HttpUrl | None = Field(
        None, description="URL del video a traducir (opcional por ahora)"
    )

    text: str | None = Field(
        None, max_length=500, description="Texto alternativo si no hay video"
    )

    language: str = Field(
        default="LSM",
        description="Lenguaje de señas (por defecto LSM - Lenguaje de Señas Mexicano)",
    )

    class Config:
        """Configuración del schema"""

        json_schema_extra = {
            "example": {"video_url": "https://example.com/video.mp4", "language": "LSM"}
        }


class TranslateResponse(BaseModel):
    translation_id: str = Field(..., description="ID único de la traducción")

    text: str = Field(..., description="Texto traducido")

    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Nivel de confianza de la traducción (0-1)"
    )

    status: str = Field(..., description="Estado: 'processing', 'completed', 'failed'")

    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Fecha y hora de creación"
    )

    class Config:
        """Configuración del schema"""

        json_schema_extra = {
            "example": {
                "translation_id": "trans_abc123",
                "text": "Hola, ¿cómo estás?",
                "confidence": 0.95,
                "status": "completed",
                "created_at": "2024-12-20T22:00:00",
            }
        }
