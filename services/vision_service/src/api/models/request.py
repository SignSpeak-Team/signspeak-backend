"""API Request Models."""

from pydantic import BaseModel, Field, field_validator


class StaticLandmarksRequest(BaseModel):
    """Static prediction: 21 hand landmarks (63 features)."""

    landmarks: list[list[float]] = Field(
        ..., min_length=21, max_length=21, description="21 landmarks [x,y,z] each"
    )
    handedness: str | None = Field(None, description="Handedness: 'Left' or 'Right'")

    @field_validator("landmarks")
    @classmethod
    def validate_coords(cls, v):
        for i, lm in enumerate(v):
            if len(lm) != 3:
                raise ValueError(f"Landmark {i}: expected 3 coords, got {len(lm)}")
        return v


class TemporalSequenceRequest(BaseModel):
    """Dynamic/Words prediction: 15 frames x 21 landmarks."""

    sequence: list[list[list[float]]] = Field(
        ...,
        min_length=15,
        max_length=15,
        description="15 frames, each with 21 landmarks",
    )
    handedness: str | None = Field(None, description="Handedness: 'Left' or 'Right'")

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v):
        for f_idx, frame in enumerate(v):
            if len(frame) != 21:
                raise ValueError(
                    f"Frame {f_idx}: expected 21 landmarks, got {len(frame)}"
                )
            for l_idx, lm in enumerate(frame):
                if len(lm) != 3:
                    raise ValueError(
                        f"Frame {f_idx}, landmark {l_idx}: expected 3 coords"
                    )
        return v


class HolisticRequest(BaseModel):
    """Holistic prediction: 226 features (pose + hands)."""

    landmarks: list[float] = Field(
        ..., min_length=226, max_length=226, description="226 holistic features"
    )


# Aliases for backward compatibility
LandmarksRequest = StaticLandmarksRequest
SequenceRequest = TemporalSequenceRequest


class LSERequest(BaseModel):
    """MSG3D prediction: 75 landmarks (Pose 33 + Hands 21*2)."""

    landmarks: list[list[float]] = Field(
        ..., min_length=75, max_length=75, description="75 landmarks [x,y,z]"
    )

    @field_validator("landmarks")
    @classmethod
    def validate_coords(cls, v):
        for i, lm in enumerate(v):
            if len(lm) != 3:
                raise ValueError(f"Landmark {i}: expected 3 coords, got {len(lm)}")
        return v
