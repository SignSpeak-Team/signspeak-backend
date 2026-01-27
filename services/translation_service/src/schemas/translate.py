from pydantic import BaseModel


class TranslateRequest(BaseModel):
    text: str | None = None
    video_url: str | None = None
    language: str = "LSM"


class TranslateResponse(BaseModel):
    text: str
    confidence: float
