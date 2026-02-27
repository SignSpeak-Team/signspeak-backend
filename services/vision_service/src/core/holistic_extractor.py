"""Holistic Landmark Extractor - 226 features (pose + both hands)."""

import mediapipe as mp
import numpy as np


class HolisticExtractor:
    """Extract holistic landmarks: pose + left hand + right hand."""

    # Indices of relevant pose landmarks (upper body only)
    POSE_INDICES = list(range(0, 25))  # Face + upper body

    def __init__(
        self,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

    def extract(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """
        Extract 226 features from RGB frame.
        Returns None if no landmarks detected.
        """
        results = self.holistic.process(frame_rgb)

        print(
            f"[DEBUG] Holistic Results: Pose={bool(results.pose_landmarks)}, Left={bool(results.left_hand_landmarks)}, Right={bool(results.right_hand_landmarks)}"
        )
        # ---------------------------

        if not self._has_valid_landmarks(results):
            print("[DEBUG] No valid landmarks found in extracted results.")
            return None

        features = []

        # Pose landmarks (~75 features: 25 landmarks × 3)
        if results.pose_landmarks:
            pose_ref = results.pose_landmarks.landmark[0]  # Nose as reference
            for idx in self.POSE_INDICES:
                lm = results.pose_landmarks.landmark[idx]
                features.extend(
                    [lm.x - pose_ref.x, lm.y - pose_ref.y, lm.z - pose_ref.z]
                )
        else:
            features.extend([0.0] * (len(self.POSE_INDICES) * 3))

        # Left hand (63 features: 21 landmarks × 3)
        features.extend(self._extract_hand(results.left_hand_landmarks))

        # Right hand (63 features: 21 landmarks × 3)
        features.extend(self._extract_hand(results.right_hand_landmarks))

        # Pad to exactly 226 if needed
        while len(features) < 226:
            features.append(0.0)

        return np.array(features[:226], dtype=np.float32)

    def _extract_hand(self, hand_landmarks) -> list:
        """Extract normalized hand landmarks."""
        if not hand_landmarks:
            return [0.0] * 63

        wrist = hand_landmarks.landmark[0]
        features = []
        for lm in hand_landmarks.landmark:
            features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        return features

    def _has_valid_landmarks(self, results) -> bool:
        """Check if we have at least pose or one hand."""
        return (
            results.pose_landmarks is not None
            or results.left_hand_landmarks is not None
            or results.right_hand_landmarks is not None
        )

    def draw(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw holistic landmarks on frame."""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS
            )
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
            )
        return frame

    def process_frame(self, frame_rgb: np.ndarray) -> tuple[np.ndarray | None, any]:
        """Process frame and return both features and raw results for drawing."""
        results = self.holistic.process(frame_rgb)
        features = None

        if self._has_valid_landmarks(results):
            features = self.extract_from_results(results)

        return features, results

    def extract_from_results(self, results) -> np.ndarray:
        """Extract features from already processed results."""
        features = []

        if results.pose_landmarks:
            pose_ref = results.pose_landmarks.landmark[0]
            for idx in self.POSE_INDICES:
                lm = results.pose_landmarks.landmark[idx]
                features.extend(
                    [lm.x - pose_ref.x, lm.y - pose_ref.y, lm.z - pose_ref.z]
                )
        else:
            features.extend([0.0] * (len(self.POSE_INDICES) * 3))

        features.extend(self._extract_hand(results.left_hand_landmarks))
        features.extend(self._extract_hand(results.right_hand_landmarks))

        while len(features) < 226:
            features.append(0.0)

        return np.array(features[:226], dtype=np.float32)

    def close(self):
        """Release resources."""
        self.holistic.close()
