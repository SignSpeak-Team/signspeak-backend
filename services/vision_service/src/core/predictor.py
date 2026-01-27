"""Sign Language Predictor - Handles all model predictions."""

import pickle
import time
from collections import deque
from typing import Any

import numpy as np
from config import (
    HOLISTIC_LABEL_ENCODER_PATH,
    HOLISTIC_MODEL_PATH,
    HOLISTIC_NUM_FEATURES,
    HOLISTIC_SEQUENCE_LENGTH,
    LABEL_ENCODER_PATH,
    LSTM_LABEL_ENCODER_PATH,
    LSTM_MODEL_PATH,
    SEQUENCE_LENGTH,
    SIGN_MODEL_PATH,
    WORDS_LABEL_ENCODER_PATH,
    WORDS_MODEL_PATH,
)
from tensorflow import keras
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential

from core.word_buffer import WordBuffer


class SignPredictor:
    """Unified predictor for LSM sign language recognition."""

    def __init__(self):
        print("[Predictor] Loading models...")
        self._load_static_model()
        self._load_dynamic_model()
        self._load_words_model()
        self._load_holistic_model()
        self._init_buffers()
        print("[Predictor] All models loaded successfully")

    # === Model Loaders ===

    def _load_static_model(self):
        """Load static alphabet model (21 letters)."""
        self.static_model = keras.models.load_model(
            str(SIGN_MODEL_PATH), compile=False, safe_mode=False
        )
        self.static_labels = self._load_encoder(LABEL_ENCODER_PATH)
        print(f"  ✓ Static: {len(self.static_labels)} letters")

    def _load_dynamic_model(self):
        """Load dynamic LSTM model (J, K, Q, X, Z, Ñ)."""
        self.lstm_model = keras.models.load_model(
            str(LSTM_MODEL_PATH), compile=False, safe_mode=False
        )
        self.lstm_labels = self._load_encoder(LSTM_LABEL_ENCODER_PATH)
        print(f"  ✓ Dynamic: {len(self.lstm_labels)} letters")

    def _load_words_model(self):
        """Load words model (249 vocabulary)."""
        self.words_model = keras.models.load_model(
            str(WORDS_MODEL_PATH), compile=False, safe_mode=False
        )
        self.words_labels = self._load_encoder(WORDS_LABEL_ENCODER_PATH)
        print(f"  ✓ Words: {len(self.words_labels)} words")

    def _load_holistic_model(self):
        """Load holistic medical model (150 words)."""
        try:
            # Try loading as full model first (standard for .h5)
            self.holistic_model = keras.models.load_model(str(HOLISTIC_MODEL_PATH))
        except Exception:
            # Fallback: Re-create architecture if file only contains weights
            print("  ! Holistic: Full load failed, rebuilding architecture...")
            self.holistic_model = Sequential(
                [
                    InputLayer(
                        input_shape=(HOLISTIC_SEQUENCE_LENGTH, HOLISTIC_NUM_FEATURES)
                    ),
                    BatchNormalization(),
                    LSTM(64, return_sequences=True),
                    Dropout(0.2),
                    LSTM(128, return_sequences=False),
                    Dropout(0.2),
                    Dense(64, activation="relu"),
                    Dropout(0.2),
                    Dense(32, activation="relu"),
                    Dense(150, activation="softmax"),
                ]
            )
            self.holistic_model.load_weights(str(HOLISTIC_MODEL_PATH))

        self.holistic_labels = self._load_encoder(HOLISTIC_LABEL_ENCODER_PATH)
        print(f"  ✓ Holistic: {len(self.holistic_labels)} medical words")

    def _load_encoder(self, path) -> dict[int, str]:
        """Load label encoder and invert it (idx -> label)."""
        with open(path, "rb") as f:
            labels = pickle.load(f)
        return {v: k for k, v in labels.items()}

    def _init_buffers(self):
        """Initialize frame buffers for sequence models."""
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.words_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.holistic_buffer = deque(maxlen=HOLISTIC_SEQUENCE_LENGTH)
        self.word_buffer = WordBuffer()

    # === Predictions ===

    def predict_static(self, landmarks: np.ndarray) -> dict[str, Any]:
        """Predict static letter from hand landmarks (63 features)."""
        self._validate_shape(landmarks, (63,))

        start = time.time()
        pred = self.static_model.predict(landmarks.reshape(1, -1), verbose=0)
        idx, conf = self._get_prediction(pred)

        return {
            "letter": self.static_labels[idx],
            "confidence": round(conf, 2),
            "type": "static",
            "processing_time_ms": round((time.time() - start) * 1000, 2),
        }

    def predict_dynamic(self, sequence: np.ndarray) -> dict[str, Any]:
        """Predict dynamic letter from sequence (15 frames x 63 features)."""
        self._validate_shape(sequence, (SEQUENCE_LENGTH, 63))

        start = time.time()
        pred = self.lstm_model.predict(
            sequence.reshape(1, SEQUENCE_LENGTH, 63), verbose=0
        )
        idx, conf = self._get_prediction(pred)

        return {
            "letter": self.lstm_labels[idx],
            "confidence": round(conf, 2),
            "type": "dynamic",
            "processing_time_ms": round((time.time() - start) * 1000, 2),
        }

    def predict_word(self, landmarks: np.ndarray) -> dict[str, Any] | None:
        """Predict word from accumulated frames (15 frames x 63 features)."""
        self._validate_shape(landmarks, (63,))
        self.words_buffer.append(landmarks)

        if len(self.words_buffer) < SEQUENCE_LENGTH:
            return None

        start = time.time()
        sequence = np.array(list(self.words_buffer))
        pred = self.words_model.predict(
            sequence.reshape(1, SEQUENCE_LENGTH, 63), verbose=0
        )
        idx, conf = self._get_prediction(pred)

        return {
            "word": self.words_labels.get(idx, "UNKNOWN"),
            "confidence": round(conf, 2),
            "type": "word",
            "processing_time_ms": round((time.time() - start) * 1000, 2),
        }

    def predict_holistic(self, landmarks: np.ndarray) -> dict[str, Any] | None:
        """Predict medical word from holistic landmarks (30 frames x 226 features)."""
        self._validate_shape(landmarks, (HOLISTIC_NUM_FEATURES,))
        self.holistic_buffer.append(landmarks)

        if len(self.holistic_buffer) < HOLISTIC_SEQUENCE_LENGTH:
            return None

        start = time.time()
        sequence = np.array(list(self.holistic_buffer))
        pred = self.holistic_model.predict(
            sequence.reshape(1, HOLISTIC_SEQUENCE_LENGTH, HOLISTIC_NUM_FEATURES),
            verbose=0,
        )
        idx, conf = self._get_prediction(pred)

        return {
            "word": self.holistic_labels.get(idx, "UNKNOWN"),
            "confidence": round(conf, 2),
            "type": "holistic_medical",
            "processing_time_ms": round((time.time() - start) * 1000, 2),
        }

    def predict_word_with_buffer(self, landmarks: np.ndarray) -> dict[str, Any] | None:
        """Predict word with repetition filtering."""
        result = self.predict_word(landmarks)
        if result is None:
            return None

        if self.word_buffer.add_detection(result["word"], result["confidence"]):
            result["phrase"] = self.word_buffer.get_phrase()
            result["accepted"] = True
            return result
        return None

    # === Utilities ===

    def _validate_shape(self, arr: np.ndarray, expected: tuple):
        if arr.shape != expected:
            raise ValueError(f"Expected {expected}, got {arr.shape}")

    def _get_prediction(self, pred: np.ndarray) -> tuple[int, float]:
        idx = int(np.argmax(pred))
        conf = float(pred[0][idx] * 100)
        return idx, conf

    def get_current_phrase(self) -> str:
        return self.word_buffer.get_phrase()

    def get_word_buffer_stats(self) -> dict:
        return self.word_buffer.get_statistics()

    def reset_buffer(self, buffer_type: str = "all"):
        """Reset specified buffer(s)."""
        buffers = {
            "letters": [self.frame_buffer],
            "words": [self.words_buffer],
            "holistic": [self.holistic_buffer],
            "word_buffer": [self.word_buffer],
            "all": [
                self.frame_buffer,
                self.words_buffer,
                self.holistic_buffer,
                self.word_buffer,
            ],
        }
        for buf in buffers.get(buffer_type, []):
            buf.clear()

    def get_models_info(self) -> dict[str, Any]:
        return {
            "static": {"count": len(self.static_labels), "type": "alphabet"},
            "dynamic": {"count": len(self.lstm_labels), "type": "alphabet"},
            "words": {"count": len(self.words_labels), "type": "vocabulary"},
            "holistic": {"count": len(self.holistic_labels), "type": "medical"},
        }


# === Singleton ===
_predictor: SignPredictor | None = None


def get_predictor() -> SignPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SignPredictor()
    return _predictor
