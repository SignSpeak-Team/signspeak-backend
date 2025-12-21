"""
API Gateway - Punto de entrada principal
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_gateway.src.settings import settings
from api_gateway.src.routes import health

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.VERSION,
    description="API Gateway para SignSpeak - Sistema de traducción de LSM",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar routers
app.include_router(health.router)


@app.on_event("startup")
async def startup_event():
    print(f"🚀 {settings.SERVICE_NAME} v{settings.VERSION} iniciando...")
    print(f"📍 Entorno: {settings.ENVIRONMENT}")
    print(f"🌐 Escuchando en: http://{settings.API_GATEWAY_HOST}:{settings.API_GATEWAY_PORT}")
    print(f"📚 Documentación: http://{settings.API_GATEWAY_HOST}:{settings.API_GATEWAY_PORT}/docs")


@app.on_event("shutdown")
async def shutdown_event():
    print(f"🛑 {settings.SERVICE_NAME} deteniendo...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.API_GATEWAY_HOST,
        port=settings.API_GATEWAY_PORT,
        reload=settings.DEBUG,
    )
