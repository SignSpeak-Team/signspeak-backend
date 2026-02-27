"""Health Check Endpoints."""

from datetime import datetime

from fastapi import APIRouter, Request
from src.services import vision_client
from src.settings import settings

router = APIRouter(
    prefix="/api/v1",
    tags=["Health"],
)


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.VERSION,
        "status": "running",
        "environment": settings.ENVIRONMENT,
        "docs": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "status": "/api/v1/status",
            "predict_static": "/api/v1/predict/static",
            "predict_dynamic": "/api/v1/predict/dynamic",
            "predict_words": "/api/v1/predict/words",
            "predict_holistic": "/api/v1/predict/holistic",
        },
    }


@router.get("/health")
async def health_check():
    """
    Simple health check for container orchestration.

    Returns minimal response for fast health checks.
    """
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "version": settings.VERSION,
    }


@router.get("/status")
async def system_status(request: Request):
    """
    Detailed system status with downstream service checks.

    Includes uptime, environment info, and health of connected services.
    """
    # Calculate real uptime
    start_time = getattr(request.app.state, "start_time", datetime.now())
    uptime = datetime.now() - start_time
    uptime_str = str(uptime).split(".")[0]  # Remove microseconds

    # Check downstream services
    services_status = {
        "api_gateway": "healthy",
        "vision_service": "unknown",
        "translation_service": "placeholder",  # Not yet integrated
    }

    # Check Vision Service health
    try:
        vision_health = await vision_client.health_check()
        services_status["vision_service"] = vision_health.get("status", "healthy")
    except Exception:
        services_status["vision_service"] = "unhealthy"

    # Determine overall status
    overall_status = (
        "operational" if services_status["vision_service"] == "healthy" else "degraded"
    )

    return {
        "status": overall_status,
        "version": f"{settings.SERVICE_NAME} {settings.VERSION}",
        "uptime": uptime_str,
        "environment": settings.ENVIRONMENT,
        "active_services": services_status,
        "timestamp": datetime.now().isoformat(),
    }
