
from fastapi import APIRouter, HTTPException, status
from schemas.translate import TranslateRequest, TranslateResponse
from datetime import datetime
import uuid

# Crear router
router = APIRouter(
    prefix="/api/v1/translate",
    tags=["Translation"],
)


@router.post("/", response_model=TranslateResponse, status_code=status.HTTP_201_CREATED)
async def translate_sign_language(request: TranslateRequest):
    if not request.video_url and not request.text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Debe proporcionar 'video_url' o 'text'"
        )

    # Generar ID único para esta traducción
    translation_id = f"trans_{uuid.uuid4().hex[:12]}"

    # SIMULACIÓN: Por ahora, solo devolvemos una respuesta de ejemplo
    # Más adelante, aquí llamaremos al ML Service

    if request.text:
        # Si enviaron texto, lo "traducimos" (simulado)
        translated_text = f"Traducción simulada de: {request.text}"
        confidence = 0.85
    else:
        # Si enviaron video, simulamos la traducción
        translated_text = "Hola, ¿cómo estás? Bienvenido a SignSpeak"
        confidence = 0.92

    # Crear respuesta
    response = TranslateResponse(
        translation_id=translation_id,
        text=translated_text,
        confidence=confidence,
        status="completed",
        created_at=datetime.now()
    )

    return response


@router.get("/{translation_id}", response_model=TranslateResponse)
async def get_translation(translation_id: str):
    if not translation_id.startswith("trans_"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ID de traducción inválido. Debe empezar con 'trans_'"
        )

    # SIMULACIÓN: Por ahora devolvemos datos de ejemplo
    # Más adelante consultaremos la base de datos

    response = TranslateResponse(
        translation_id=translation_id,
        text="Esta es una traducción de ejemplo",
        confidence=0.90,
        status="completed",
        created_at=datetime.now()
    )

    return response


@router.get("/", response_model=list[TranslateResponse])
async def list_translations(limit: int = 10, skip: int = 0):
    if limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El límite máximo es 100"
        )

    # SIMULACIÓN: Devolvemos una lista de ejemplo
    translations = []

    for i in range(min(limit, 3)):  # Solo 3 ejemplos por ahora
        translations.append(
            TranslateResponse(
                translation_id=f"trans_{uuid.uuid4().hex[:12]}",
                text=f"Traducción de ejemplo #{i + 1}",
                confidence=0.88 + (i * 0.02),
                status="completed",
                created_at=datetime.now()
            )
        )
    return translations