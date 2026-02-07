"""Post-processing for video sequence predictions."""

from dataclasses import dataclass
from typing import Any

from config import DUPLICATE_TIME_THRESHOLD, MIN_WINDOW_CONFIDENCE


@dataclass
class DetectedSegment:
    """Represents a detected word segment in video."""

    word: str
    start_time: float
    end_time: float
    confidence: float


class SequenceProcessor:
    """Post-processes video prediction sequences for better accuracy."""

    def __init__(
        self,
        min_confidence: float = MIN_WINDOW_CONFIDENCE,
        duplicate_threshold: float = DUPLICATE_TIME_THRESHOLD,
    ):
        """
        Initialize sequence processor.

        Args:
            min_confidence: Minimum confidence to accept prediction (0-100)
            duplicate_threshold: Time window (seconds) to merge duplicates
        """
        self.min_confidence = min_confidence
        self.duplicate_threshold = duplicate_threshold

    def process_segments(
        self, raw_segments: list[dict[str, Any]]
    ) -> tuple[list[DetectedSegment], dict[str, Any]]:
        """
        Process raw prediction segments with filtering and merging.

        Args:
            raw_segments: List of dicts with word, start_time, end_time, confidence

        Returns:
            Tuple of (processed_segments, statistics)
        """
        # Step 1: Filter by confidence
        confident_segments = self._filter_by_confidence(raw_segments)

        # Step 2: Remove "UNKNOWN" predictions
        valid_segments = [s for s in confident_segments if s["word"] != "UNKNOWN"]

        # Step 3: Merge temporal duplicates
        merged_segments = self._merge_duplicates(valid_segments)

        # Convert to dataclass
        final_segments = [
            DetectedSegment(
                word=s["word"],
                start_time=s["start_time"],
                end_time=s["end_time"],
                confidence=s["confidence"],
            )
            for s in merged_segments
        ]

        # Calculate statistics
        stats = self._calculate_stats(raw_segments, final_segments)

        return final_segments, stats

    def _filter_by_confidence(
        self, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter segments below minimum confidence threshold."""
        return [s for s in segments if s["confidence"] >= self.min_confidence]

    def _merge_duplicates(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Merge consecutive segments with same word within time threshold.

        This handles cases where the same sign is detected across overlapping windows.
        """
        if not segments:
            return []

        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x["start_time"])

        merged = []
        current = sorted_segments[0].copy()

        for next_seg in sorted_segments[1:]:
            # Check if same word and within time threshold
            time_gap = next_seg["start_time"] - current["end_time"]

            if (
                next_seg["word"] == current["word"]
                and time_gap <= self.duplicate_threshold
            ):
                # Merge: extend end time and take higher confidence
                current["end_time"] = max(current["end_time"], next_seg["end_time"])
                current["confidence"] = max(current["confidence"], next_seg["confidence"])
            else:
                # Different word or too far apart - save current and start new
                merged.append(current)
                current = next_seg.copy()

        # Don't forget the last one
        merged.append(current)

        return merged

    def _calculate_stats(
        self, raw: list[dict[str, Any]], final: list[DetectedSegment]
    ) -> dict[str, Any]:
        """Calculate processing statistics."""
        total_windows = len(raw)
        detected_before_filter = len([s for s in raw if s["word"] != "UNKNOWN"])
        filtered_words = len(final)

        avg_confidence = 0.0
        if final:
            avg_confidence = sum(s.confidence for s in final) / len(final)

        return {
            "total_windows": total_windows,
            "detected_words": detected_before_filter,
            "filtered_words": filtered_words,
            "average_confidence": round(avg_confidence, 2),
            "filter_rate": round(
                (1 - filtered_words / detected_before_filter) * 100
                if detected_before_filter > 0
                else 0,
                2,
            ),
        }

    def build_phrase(self, segments: list[DetectedSegment]) -> str:
        """Build phrase from detected segments."""
        return " ".join(s.word for s in segments)
