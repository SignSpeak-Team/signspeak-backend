import httpx
from src.settings import settings

async def translate(payload: dict) -> dict:
    url = f"{settings.TRANSLATION_SERVICE_URL}/api/v1/translate/"

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
