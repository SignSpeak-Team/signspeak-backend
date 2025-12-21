"""
Health Check Endpoints
"""
from fastapi import APIRouter


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