"""Video Processing Pipeline."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
from config import HOLISTIC_NUM_FEATURES

from core.holistic_extractor import HolisticExtractor


class VideoProcessor:
    """Manages video file processing and feature extraction."""

    def __init__(self):
        # MediaPipe extractor is expensive, initialize once if possible or per request
        self.extractor = HolisticExtractor(static_image_mode=False)
        self.image_extractor = HolisticExtractor(
            static_image_mode=True,
            min_detection_confidence=0.1,  # Very sensitive for static images
            min_tracking_confidence=0.1,
        )

    def process_video_bytes(
        self, video_bytes: bytes, target_frames: int = 30
    ) -> np.ndarray:
        """Process video bytes, resizing and extracting holistic features."""
        # Save bytes to temp file because cv2.VideoCapture needs a path
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        try:
            return self._process_file(tmp_path, target_frames)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _process_file(self, video_path: Path, target_frames: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            # Fallback if frame count is not available
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            total_frames = len(frames)
            # Re-open or just use the list if memory permits, but for large videos better re-open or optimize
            # For simplicity, if we read them all, uses them.
            # But let's stick to reading by index logic if total_frames is valid.
            if total_frames == 0:
                raise ValueError("Could not read video frames")

        # Select indices uniformly
        if total_frames <= target_frames:
            indices = np.arange(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

        frames_features = []
        current_idx = 0

        # Reset cap if we read it all (in fallback case) or just start from 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while cap.isOpened() and len(frames_features) < target_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if current_idx in indices:
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                features = self.extractor.extract(frame_rgb)

                # Handle missing detection (pad with zeros)
                if features is None:
                    features = np.zeros(HOLISTIC_NUM_FEATURES, dtype=np.float32)

                frames_features.append(features)

            current_idx += 1

        cap.release()

        # Final padding if video was too short
        while len(frames_features) < target_frames:
            frames_features.append(np.zeros(HOLISTIC_NUM_FEATURES, dtype=np.float32))

        return np.array(frames_features, dtype=np.float32)

    def process_video_sliding_window(
        self,
        video_bytes: bytes,
        window_size_sec: float = 2.0,  # Updated: better for sign language
        stride_sec: float = 0.75,  # Updated: 50% overlap for transitions
        target_frames: int = 30,
    ) -> list[dict]:
        """
        Process video continuously (linear scan) and apply sliding window on extracted features.
        This preserves MediaPipe tracking context and is much faster.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        try:
            # 1. Extract features for the ENTRIE video sequentially
            all_features, fps = self._process_entire_video(tmp_path)
            total_frames = len(all_features)

            if total_frames == 0 or fps <= 0:
                return []

            segments = []
            window_frames = int(window_size_sec * fps)
            stride_frames = int(stride_sec * fps)

            # 2. Apply sliding window over the PRE-CALCULATED features
            for start_frame in range(0, total_frames, stride_frames):
                end_frame = min(start_frame + window_frames, total_frames)

                # Skip windows that are too short (less than 75% filled)
                if (end_frame - start_frame) < (window_frames * 0.75):
                    continue

                # Get slice
                raw_window = all_features[start_frame:end_frame]

                # Resample to target_frames (30)
                processed_window = self._resample_sequence(raw_window, target_frames)

                segments.append(
                    {
                        "start_time": start_frame / fps,
                        "end_time": end_frame / fps,
                        "features": processed_window,
                    }
                )

            return segments

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def process_image(self, image_bytes: bytes) -> np.ndarray | None:
        """Process a single image and extract holistic features."""
        # Decode bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Could not decode image")

        print(f"[DEBUG] Processing image of shape: {frame.shape}")

        # Convert to RGB (MediaPipe requirement)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract using the static mode extractor
        features = self.image_extractor.extract(frame_rgb)

        if features is None:
            print("[DEBUG] Extractor returned None (No landmarks found)")
        else:
            print(f"[DEBUG] Extractor returned features shape: {features.shape}")

        return features

    def _process_entire_video(self, video_path: Path) -> tuple[np.ndarray, float]:
        """Read video strictly sequentially to keep MediaPipe happy."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        all_features = []

        # Reset extractor for new video context (crucial for tracking)
        # self.extractor.reset() # If extractor supported it, otherwise we rely on continuity

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features = self.extractor.extract(frame_rgb)

            if features is None:
                features = np.zeros(HOLISTIC_NUM_FEATURES, dtype=np.float32)

            all_features.append(features)

        cap.release()
        return np.array(all_features, dtype=np.float32), fps

    def _resample_sequence(
        self, sequence: np.ndarray, target_length: int
    ) -> np.ndarray:
        """Resample a sequence of N frames to `target_length` frames using uniform sampling."""
        current_length = len(sequence)

        if current_length == target_length:
            return sequence

        if current_length == 0:
            return np.zeros((target_length, HOLISTIC_NUM_FEATURES), dtype=np.float32)

        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        return sequence[indices]

    def close(self):
        self.extractor.close()
