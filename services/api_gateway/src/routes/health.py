"""
Health Check Endpoints
"""

from datetime import datetime, timedelta

from fastapi import APIRouter
from src.settings import settings

# Crear router
router = APIRouter(
    prefix="/api/v1",
    tags=["Health"],
)


@router.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "service": "API Gateway",
        "version": "1.0.0",
        "status": "running",
        "environment": "development",
        "docs": "/docs",
    }


@router.get("/health")
async def health_check():
    """Health Check"""
    return {
        "status": "healthy",
        "service": "API Gateway",
        "version": "1.0.0",
    }


@router.get("/status")
async def system_status():
    uptime_seconds = 7200
    uptime = str(timedelta(seconds=uptime_seconds))

    return {
        "status": "operational",
        "version": settings.SERVICE_NAME + " " + settings.VERSION,
        "uptime": uptime,
        "environment": settings.ENVIRONMENT,
        "total_translations": 42,  # Simulado - luego será de BD
        "active_services": {
            "api_gateway": "healthy",
            "translation_service": "not_implemented",
            "ml_service": "not_implemented",
            "storage_service": "not_implemented",
        },
        "timestamp": datetime.now().isoformat(),
    }
