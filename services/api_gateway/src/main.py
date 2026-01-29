from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routes import health, prediction, translate
from src.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - track start time for uptime calculation
    app.state.start_time = datetime.now()
    print(f"🚀 {settings.SERVICE_NAME} v{settings.VERSION} iniciando...")
    print(f"📍 Entorno: {settings.ENVIRONMENT}")
    print(
        f"🌐 Escuchando en: http://{settings.API_GATEWAY_HOST}:{settings.API_GATEWAY_PORT}"
    )
    print(
        f"📚 Documentación: http://{settings.API_GATEWAY_HOST}:{settings.API_GATEWAY_PORT}/docs"
    )
    print(f"🎯 Vision Service: {settings.VISION_SERVICE_URL}")
    yield
    # Shutdown
    print(f"🛑 {settings.SERVICE_NAME} detenido")


app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.VERSION,
    description="API Gateway para SignSpeak - Sistema de traducción de LSM",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(translate.router)
app.include_router(prediction.router)

