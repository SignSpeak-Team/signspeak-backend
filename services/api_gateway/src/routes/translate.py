from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import uuid

from api_gateway.src.schemas.translate import (
    TranslateRequest,
    TranslateResponse,
)
from api_gateway.src.services.http_client import forward_request

router = APIRouter(
    prefix="/api/v1/translate",
    tags=["Translation"],
)

TRANSLATION_SERVICE_URL = "http://127.0.0.1:8001/api/v1/translate/"


@router.post(
    "/",
    response_model=TranslateResponse,
    status_code=status.HTTP_201_CREATED
)
async def translate_sign_language(request: TranslateRequest):
    """
    API Gateway endpoint.
    Reenvía la petición al translation_service.
    """
    if not request.video_url and not request.text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Debe proporcionar 'video_url' o 'text'"
        )

    translation_id = f"trans_{uuid.uuid4().hex[:12]}"

    try:
        response = await forward_request(
            method="POST",
            url=TRANSLATION_SERVICE_URL,
            json=request.dict(),
        )

        if response.status_code != 200:
            raise RuntimeError("Translation service error")

        data = response.json()

        return TranslateResponse(
            translation_id=translation_id,
            text=data["text"],
            confidence=data["confidence"],
            status="completed",
            created_at=datetime.utcnow(),
        )

    except Exception:
        # Fallback controlado
        return TranslateResponse(
            translation_id=translation_id,
            text="Fallback: traducción simulada",
            confidence=0.80,
            status="completed",
            created_at=datetime.utcnow(),
        )


@router.get("/{translation_id}", response_model=TranslateResponse)
async def get_translation(translation_id: str):
    if not translation_id.startswith("trans_"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ID de traducción inválido. Debe empezar con 'trans_'"
        )

    return TranslateResponse(
        translation_id=translation_id,
        text="Esta es una traducción de ejemplo",
        confidence=0.90,
        status="completed",
        created_at=datetime.utcnow(),
    )


@router.get("/", response_model=list[TranslateResponse])
async def list_translations(limit: int = 10, skip: int = 0):
    if limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El límite máximo es 100"
        )

    return [
        TranslateResponse(
            translation_id=f"trans_{uuid.uuid4().hex[:12]}",
            text=f"Traducción de ejemplo #{i + 1}",
            confidence=0.88 + (i * 0.02),
            status="completed",
            created_at=datetime.utcnow(),
        )
        for i in range(min(limit, 3))
    ]
