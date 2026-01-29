from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routes.translate import router as translate_router
from src.settings import settings

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.VERSION,
    description="Translation Service - Orquestación de traducciones LSM",
)

# CORS para permitir comunicación entre servicios
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(translate_router)
