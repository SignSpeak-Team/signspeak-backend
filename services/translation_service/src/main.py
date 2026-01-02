from fastapi import FastAPI
from src.routes.translate import router as translate_router


app = FastAPI(
    title="Translation Service",
    version="1.0",
)

app.include_router(translate_router)
