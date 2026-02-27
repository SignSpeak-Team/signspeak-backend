"""Media Processing Endpoints - Video/Image upload and translation."""

import time

import numpy as np
from config import DEFAULT_STRIDE_SEC, DEFAULT_WINDOW_SIZE_SEC, MIN_WINDOW_CONFIDENCE
from core.predictor import SignPredictor, get_predictor
from core.sequence_processor import SequenceProcessor
from core.video_processor import VideoProcessor
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.models.response import DetectionStats, VideoSegment, VideoTranslationResponse

router = APIRouter(prefix="/media", tags=["Media Processing"])

# Singleton para video processor
_video_processor: VideoProcessor | None = None


def get_video_processor() -> VideoProcessor:
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor


@router.post("/translate/video", response_model=VideoTranslationResponse)
async def translate_video(
    file: UploadFile = File(..., description="Video file (MP4, MOV, AVI)"),
    mode: str = Form(
        "continuous",
        description="Mode: 'continuous' (default, detects sequences) or 'holistic' (single word)",
    ),
    min_confidence: float = Form(
        MIN_WINDOW_CONFIDENCE,
        description="Minimum confidence threshold (0-100) for continuous mode",
        ge=0,
        le=100,
    ),
    window_size: float = Form(
        DEFAULT_WINDOW_SIZE_SEC,
        description="Window size in seconds for continuous mode",
        gt=0,
        le=5.0,
    ),
    stride: float = Form(
        DEFAULT_STRIDE_SEC,
        description="Stride in seconds for continuous mode",
        gt=0,
        le=3.0,
    ),
    predictor: SignPredictor = Depends(get_predictor),
    processor: VideoProcessor = Depends(get_video_processor),
):
    """
    Translate sign language video to text.

    **Modes:**
    - **continuous** (default): Detects sequences of words using sliding window
    - **holistic**: Processes entire video as single word

    **Parameters (continuous mode only):**
    - **min_confidence**: Filter predictions below this threshold
    - **window_size**: Size of sliding window (seconds)
    - **stride**: Overlap between windows (seconds)
    """
    start_time = time.time()

    # Validate file type
    allowed_types = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            400, f"Invalid file type {file.content_type}. Allowed: {allowed_types}"
        )

    # Read file
    try:
        video_bytes = await file.read()
    except Exception as e:
        raise HTTPException(500, "Could not read file") from e

    # Check file size (50MB max)
    if len(video_bytes) > 50 * 1024 * 1024:
        raise HTTPException(400, "File too large. Max size: 50MB")

    try:
        if mode == "continuous":
            # --- Continuous Mode (Sequence Detection) ---
            extraction_start = time.time()
            segments_data = processor.process_video_sliding_window(
                video_bytes, window_size_sec=window_size, stride_sec=stride
            )
            extraction_time = (time.time() - extraction_start) * 1000

            # Predict for each window
            prediction_start = time.time()
            raw_detections = []

            for seg in segments_data:
                res = predictor.predict_holistic_sequence(seg["features"])
                raw_detections.append(
                    {
                        "word": res["word"],
                        "start_time": seg["start_time"],
                        "end_time": seg["end_time"],
                        "confidence": res["confidence"],
                    }
                )

            prediction_time = (time.time() - prediction_start) * 1000

            # Post-process with SequenceProcessor
            seq_processor = SequenceProcessor(min_confidence=min_confidence)
            processed_segments, stats = seq_processor.process_segments(raw_detections)

            # Build final phrase
            full_phrase = seq_processor.build_phrase(processed_segments)

            # Calculate aggregate confidence
            avg_confidence = stats["average_confidence"]

            return VideoTranslationResponse(
                word=full_phrase if full_phrase else "[No words detected]",
                confidence=avg_confidence,
                extraction_time_ms=round(extraction_time, 2),
                prediction_time_ms=round(prediction_time, 2),
                total_time_ms=round((time.time() - start_time) * 1000, 2),
                frames_processed=len(segments_data) * 30,
                segments=[
                    VideoSegment(
                        word=s.word,
                        start_time=round(s.start_time, 2),
                        end_time=round(s.end_time, 2),
                        confidence=s.confidence,
                    )
                    for s in processed_segments
                ],
                detection_stats=DetectionStats(**stats),
            )

        else:
            # --- Holistic Mode (Default) ---
            extraction_start = time.time()
            sequence = processor.process_video_bytes(video_bytes, target_frames=30)
            extraction_time = (time.time() - extraction_start) * 1000

            # Run prediction
            prediction_start = time.time()
            result = predictor.predict_holistic_sequence(sequence)
            prediction_time = (time.time() - prediction_start) * 1000

            total_time = (time.time() - start_time) * 1000

            return VideoTranslationResponse(
                word=result["word"],
                confidence=result["confidence"],
                extraction_time_ms=round(extraction_time, 2),
                prediction_time_ms=round(prediction_time, 2),
                total_time_ms=round(total_time, 2),
                frames_processed=30,
                segments=[],
            )

    except Exception as e:
        import logging

        logging.error(f"Error processing video: {str(e)}")
        raise HTTPException(500, f"Error processing video: {str(e)}")


@router.post("/translate/image", response_model=VideoTranslationResponse)
async def translate_image(
    file: UploadFile = File(..., description="Image file (JPG, PNG)"),
    predictor: SignPredictor = Depends(get_predictor),
    processor: VideoProcessor = Depends(get_video_processor),
):
    """
    Translate sign language image to letter (Static Alphabet).

    Detects hand landmarks and predicts the letter (A-Y).
    Prioritizes Right Hand, then Left Hand.
    """
    start_time = time.time()

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    try:
        content = await file.read()
        features = processor.process_image(content)

        if features is None:
            raise HTTPException(400, "No landmarks detected in image")

        # Indices from HolisticExtractor
        # Pose: 0-75 (25*3)
        # Left Hand: 75-138 (21*3)
        # Right Hand: 138-201 (21*3)

        left_hand = features[75:138]
        right_hand = features[138:201]

        # Determine which hand to use
        # Check if non-zero (sum abs > epsilon)
        target_hand = None

        if np.sum(np.abs(right_hand)) > 0.1:
            target_hand = right_hand
        elif np.sum(np.abs(left_hand)) > 0.1:
            target_hand = left_hand
        else:
            raise HTTPException(400, "No hand landmarks detected")

        # Predict
        result = predictor.predict_static(target_hand)

        return VideoTranslationResponse(
            word=result["letter"],
            confidence=result["confidence"],
            extraction_time_ms=0,
            prediction_time_ms=result["processing_time_ms"],
            total_time_ms=round((time.time() - start_time) * 1000, 2),
            frames_processed=1,
            segments=[],
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, f"Image processing failed: {str(e)}")
