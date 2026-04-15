import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from api.routes import health, media, prediction


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events — pre-loads ML models on startup."""
    print("=" * 60)
    print("SignSpeak Vision API - Starting")
    print("=" * 60)
    print("Loading ML models...")

    from core.predictor import get_predictor

    try:
        get_predictor()
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")

    print("=" * 60)
    print("API ready!")
    print("=" * 60)

    yield

    print("SignSpeak Vision API - Shutting down")


# Crear aplicación FastAPI con lifespan registrado
app = FastAPI(
    title="SignSpeak Vision API",
    description="API REST para reconocimiento de lenguaje de señas mexicano (LSM)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# CORS — leer del entorno en producción
_raw_origins = os.getenv("CORS_ORIGINS", "*")
_origins = (
    [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(prediction.router, prefix="/api/v1")
app.include_router(media.router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz - Información básica de la API"""
    return {
        "service": "SignSpeak Vision API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
