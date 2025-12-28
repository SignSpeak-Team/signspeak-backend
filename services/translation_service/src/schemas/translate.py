from pydantic import BaseModel
from typing import Optional

class TranslateRequest(BaseModel):
    text: Optional[str] = None
    video_url: Optional[str] = None
    language: str = "LSM"

class TranslateResponse(BaseModel):
    text: str
    confidence: float



