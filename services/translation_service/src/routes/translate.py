from fastapi import APIRouter, HTTPException
from translation_service.src.schemas.translate import (
    TranslateRequest,
    TranslateResponse
)

router = APIRouter(
    prefix="/api/v1/translate",
    tags=["Translation Service"]
)


@router.post("/", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    if not request.text and not request.video_url:
        raise HTTPException(
            status_code=422,
            detail="Debes enviar text o video_url",
        )

    if request.text:
        return TranslateResponse(
            text=f"[ML] Traducción de: {request.text}",
            confidence=0.93
        )

    return TranslateResponse(
        text="[ML] Hola, ¿cómo estás?",
        confidence=0.95
    )
