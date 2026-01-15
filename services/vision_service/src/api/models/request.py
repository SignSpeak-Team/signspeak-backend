"""Modelos Pydantic para las requests de la API"""

from pydantic import BaseModel, Field, validator
from typing import List


class LandmarksRequest(BaseModel):
    """Request para predicción estática con landmarks"""
    
    landmarks: List[List[float]] = Field(
        ...,
        description="21 hand landmarks, cada uno con [x, y, z] coordenadas",
        min_length=21,
        max_length=21
    )
    
    @validator('landmarks')
    def validate_landmarks_format(cls, v):
        """Validar que cada landmark tenga 3 coordenadas"""
        for i, landmark in enumerate(v):
            if len(landmark) != 3:
                raise ValueError(f"Landmark {i} debe tener 3 coordenadas [x, y, z], tiene {len(landmark)}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "landmarks": [
                    [0.5, 0.3, 0.1],
                    [0.52, 0.31, 0.11],
                    # ... (21 total)
                ]
            }
        }


class SequenceRequest(BaseModel):
    """Request para predicción dinámica con secuencia de frames"""
    
    sequence: List[List[List[float]]] = Field(
        ...,
        description="Secuencia de 15 frames, cada frame con 21 landmarks de 3 coordenadas",
        min_length=15,
        max_length=15
    )
    
    @validator('sequence')
    def validate_sequence_format(cls, v):
        """Validar formato de la secuencia"""
        for frame_idx, frame in enumerate(v):
            if len(frame) != 21:
                raise ValueError(f"Frame {frame_idx} debe tener 21 landmarks, tiene {len(frame)}")
            for landmark_idx, landmark in enumerate(frame):
                if len(landmark) != 3:
                    raise ValueError(
                        f"Frame {frame_idx}, landmark {landmark_idx} debe tener 3 coords, tiene {len(landmark)}"
                    )
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "sequence": [
                    [[0.5, 0.3, 0.1], [0.52, 0.31, 0.11]],  # Frame 1 (simplificado)
                    # ... (15 frames total)
                ]
            }
        }
