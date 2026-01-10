"""
FastAPI Application - SignSpeak Vision Service
API REST para reconocimiento de lenguaje de señas LSM
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import health, prediction

# Crear aplicación FastAPI
app = FastAPI(
    title="SignSpeak Vision API",
    description="API REST para reconocimiento de lenguaje de señas mexicano (LSM)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS (permitir requests desde frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(prediction.router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """
    Endpoint raíz - Información básica de la API
    """
    return {
        "service": "SignSpeak Vision API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.on_event("startup")
async def startup_event():
    """Evento al iniciar la aplicación"""
    print("=" * 60)
    print("SignSpeak Vision API - Starting")
    print("=" * 60)
    print("Loading ML models...")
    
    # Pre-cargar modelos (carga lazy, solo cuando se hace first request)
    from core.predictor import get_predictor
    try:
        predictor = get_predictor()
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
    
    print("=" * 60)
    print("API ready!")
    print("Docs: http://localhost:8000/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Evento al cerrar la aplicación"""
    print("SignSpeak Vision API - Shutting down")
