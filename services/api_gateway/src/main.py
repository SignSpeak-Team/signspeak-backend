from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api_gateway.src.settings import settings
from api_gateway.src.routes import health, translate

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 {settings.SERVICE_NAME} v{settings.VERSION} iniciando...")
    print(f"📍 Entorno: {settings.ENVIRONMENT}")
    print(
        f"🌐 Escuchando en: http://{settings.API_GATEWAY_HOST}:{settings.API_GATEWAY_PORT}"
    )
    print(
        f"📚 Documentación: http://{settings.API_GATEWAY_HOST}:{settings.API_GATEWAY_PORT}/docs"
    )
    yield

    print(f"🛑 {settings.SERVICE_NAME} deteniendo...")

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.VERSION,
    description="API Gateway para SignSpeak - Sistema de traduccion de LSM",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#REGISTRAR ROUTERS

app.include_router(health.router)
app.include_router(translate.router)