"""Modelos Pydantic para las responses de la API"""

from pydantic import BaseModel, Field
from typing import Literal, Dict, List
from datetime import datetime


class PredictionResponse(BaseModel):
    """Response estándar para predicciones"""
    
    letter: str = Field(..., description="Letra predicha")
    confidence: float = Field(..., description="Confianza de la predicción (0-100)")
    type: Literal["static", "dynamic"] = Field(..., description="Tipo de predicción")
    processing_time_ms: float = Field(..., description="Tiempo de procesamiento en milisegundos")
    
    class Config:
        json_schema_extra = {
            "example": {
                "letter": "A",
                "confidence": 95.8,
                "type": "static",
                "processing_time_ms": 12.5
            }
        }


class HealthResponse(BaseModel):
    """Response para health check"""
    
    status: Literal["healthy", "unhealthy"] = Field(..., description="Estado del servicio")
    version: str = Field(..., description="Versión de la API")
    models_loaded: bool = Field(..., description="Si los modelos ML están cargados")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp de la respuesta")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models_loaded": True,
                "timestamp": "2026-01-10T15:30:00Z"
            }
        }


class ModelInfo(BaseModel):
    """Información de un modelo ML"""
    
    letters: List[str] = Field(..., description="Letras que puede predecir")
    count: int = Field(..., description="Número de letras")
    accuracy: float = Field(..., description="Accuracy del modelo (%)")
    version: str = Field(..., description="Versión del modelo")


class ModelsInfoResponse(BaseModel):
    """Response con información de todos los modelos"""
    
    static_model: ModelInfo
    dynamic_model: ModelInfo
    
    class Config:
        json_schema_extra = {
            "example": {
                "static_model": {
                    "letters": ["A", "B", "C"],
                    "count": 21,
                    "accuracy": 95.2,
                    "version": "2.0"
                },
                "dynamic_model": {
                    "letters": ["J", "K", "Ñ", "Q", "X", "Z"],
                    "count": 6,
                    "accuracy": 99.79,
                    "version": "2.0"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Response estándar para errores"""
    
    detail: str = Field(..., description="Descripción del error")
    error_code: str = Field(..., description="Código del error")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Invalid landmarks format",
                "error_code": "INVALID_INPUT"
            }
        }


class WordPredictionResponse(BaseModel):
    """Response para predicción de palabras con buffer"""
    
    word: str = Field(..., description="Palabra predicha")
    confidence: float = Field(..., description="Confianza de la predicción (0-100)")
    phrase: str = Field(..., description="Frase acumulada en el buffer")
    accepted: bool = Field(..., description="Si la palabra fue aceptada por el buffer")
    processing_time_ms: float = Field(..., description="Tiempo de procesamiento en ms")
    
    class Config:
        json_schema_extra = {
            "example": {
                "word": "hola",
                "confidence": 92.5,
                "phrase": "buenos días hola",
                "accepted": True,
                "processing_time_ms": 45.2
            }
        }


class BufferStatsResponse(BaseModel):
    """Response con estadísticas del buffer de palabras"""
    
    total_received: int = Field(..., description="Total de detecciones recibidas")
    total_accepted: int = Field(..., description="Total de detecciones aceptadas")
    rejected_by_cooldown: int = Field(..., description="Rechazadas por cooldown")
    rejected_by_confidence: int = Field(..., description="Rechazadas por baja confianza")
    acceptance_rate: float = Field(..., description="Tasa de aceptación (%)")
    current_phrase: str = Field(..., description="Frase actual en el buffer")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_received": 50,
                "total_accepted": 12,
                "rejected_by_cooldown": 30,
                "rejected_by_confidence": 8,
                "acceptance_rate": 24.0,
                "current_phrase": "hola buenos días"
            }
        }

