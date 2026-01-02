from fastapi import APIRouter, HTTPException, status
from src.schemas.translate import TranslateRequest, TranslateResponse
from src.services.translation_client import translate

router = APIRouter(
    prefix="/api/v1/translate",
    tags=["Translation"],
)

@router.post("/", response_model=TranslateResponse)
async def translate_sign_language(request: TranslateRequest):
    try:
        result = await translate(request.model_dump(exclude_none=True))

        return TranslateResponse(
            translation_id="trans_gateway",
            text=result["text"],
            confidence=result["confidence"],
            status="completed",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        )
