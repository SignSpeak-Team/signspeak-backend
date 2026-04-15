"""Sign Language Predictor - Handles all model predictions."""

import pickle
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from config import (
    HF_MODEL_REPO,
    HF_TOKEN,
    HOLISTIC_NUM_FEATURES,
    HOLISTIC_SEQUENCE_LENGTH,
    MODELS_DIR,
    SEQUENCE_LENGTH,
)
from tensorflow import keras
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential

from core.metrics import PREDICTION_CONFIDENCE, PREDICTION_LATENCY, PREDICTIONS_TOTAL
from core.msg3d_predictor import MSG3DPredictor
from core.word_buffer import WordBuffer


def _resolve_models_dir() -> Path:
    """Return local models dir if it exists, otherwise download from HF Hub."""
    if MODELS_DIR.exists() and any(MODELS_DIR.iterdir()):
        print(f"[Predictor] Using local models from {MODELS_DIR}")
        return MODELS_DIR

    print(
        f"[Predictor] Local models not found. Downloading from HF Hub: {HF_MODEL_REPO}"
    )
    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=HF_MODEL_REPO,
            token=HF_TOKEN,
            repo_type="model",
            local_dir=str(MODELS_DIR),
        )
        print(f"[Predictor] Models downloaded to {local_dir}")
        return Path(local_dir)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download models from HF Hub '{HF_MODEL_REPO}': {e}. "
            "Set HF_MODEL_REPO env var or provide a local models/ directory."
        ) from e


class SignPredictor:
    """Unified predictor for LSM sign language recognition."""

    def __init__(self):
        print("[Predictor] Loading models...")
        # Resolve models dir: local fallback → HF Hub download
        _resolved = _resolve_models_dir()
        self._models_dir = _resolved
        self._load_static_model()
        self._load_dynamic_model()
        self._load_words_model()
        self._load_holistic_model()
        self._load_msg3d_model()
        self._init_buffers()
        print("[Predictor] All models loaded successfully")

    # === Model Loaders ===

    def _load_static_model(self):
        """Load static alphabet model (21 letters)."""
        self.static_model = keras.models.load_model(
            str(self._models_dir / "sign_model.keras"), compile=False, safe_mode=False
        )
        self.static_labels = self._load_encoder(self._models_dir / "label_encoder.pkl")
        print(f"  ✓ Static: {len(self.static_labels)} letters")

    def _load_dynamic_model(self):
        """Load dynamic LSTM model (J, K, Q, X, Z, Ñ)."""
        self.lstm_model = keras.models.load_model(
            str(self._models_dir / "lstm_letters.keras"), compile=False, safe_mode=False
        )
        self.lstm_labels = self._load_encoder(
            self._models_dir / "lstm_label_encoder.pkl"
        )
        print(f"  ✓ Dynamic: {len(self.lstm_labels)} letters")

    def _load_words_model(self):
        """Load words model (249 vocabulary)."""
        self.words_model = keras.models.load_model(
            str(self._models_dir / "words_model.keras"), compile=False, safe_mode=False
        )
        self.words_labels = self._load_encoder(
            self._models_dir / "words_label_encoder.pkl"
        )
        print(f"  ✓ Words: {len(self.words_labels)} words")

    def _load_holistic_model(self):
        """Load holistic medical model (150 words)."""
        holistic_path = self._models_dir / "best_model.h5"
        try:
            # Try loading as full model first (standard for .h5)
            self.holistic_model = keras.models.load_model(str(holistic_path))
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
            self.holistic_model.load_weights(str(holistic_path))

        self.holistic_labels = self._load_encoder(
            self._models_dir / "holistic_label_encoder.pkl"
        )
        print(f"  ✓ Holistic: {len(self.holistic_labels)} medical words")

    def _load_msg3d_model(self):
        """Load MSG3D LSE model (300 medical signs)."""
        self.msg3d_predictor = MSG3DPredictor()

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
        self.lse_buffer = deque(maxlen=64)  # MSG3D sequence length
        self.word_buffer = WordBuffer()

    # === Predictions ===

    def predict_static(self, landmarks: np.ndarray) -> dict[str, Any]:
        """Predict static letter from hand landmarks (63 features)."""
        self._validate_shape(landmarks, (63,))

        start = time.time()
        pred = self.static_model.predict(landmarks.reshape(1, -1), verbose=0)
        idx, conf = self._get_prediction(pred)

        # Record metrics
        PREDICTION_LATENCY.labels(model_type="static").observe(time.time() - start)
        PREDICTIONS_TOTAL.labels(model_type="static", status="success").inc()
        PREDICTION_CONFIDENCE.labels(model_type="static").observe(conf)

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

        # Record metrics
        PREDICTION_LATENCY.labels(model_type="dynamic").observe(time.time() - start)
        PREDICTIONS_TOTAL.labels(model_type="dynamic", status="success").inc()
        PREDICTION_CONFIDENCE.labels(model_type="dynamic").observe(conf)

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

        # Record metrics
        PREDICTION_LATENCY.labels(model_type="word").observe(time.time() - start)
        PREDICTIONS_TOTAL.labels(model_type="word", status="success").inc()
        PREDICTION_CONFIDENCE.labels(model_type="word").observe(conf)

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

        # Record metrics
        PREDICTION_LATENCY.labels(model_type="holistic").observe(time.time() - start)
        PREDICTIONS_TOTAL.labels(model_type="holistic", status="success").inc()
        PREDICTION_CONFIDENCE.labels(model_type="holistic").observe(conf)

        return {
            "word": self.holistic_labels.get(idx, "UNKNOWN"),
            "confidence": round(conf, 2),
            "type": "holistic_medical",
            "processing_time_ms": round((time.time() - start) * 1000, 2),
        }

    def predict_holistic_sequence(self, sequence: np.ndarray) -> dict[str, Any]:
        """Predict from complete holistic sequence (30 frames × 226 features)."""
        self._validate_shape(
            sequence, (HOLISTIC_SEQUENCE_LENGTH, HOLISTIC_NUM_FEATURES)
        )

        start = time.time()
        pred = self.holistic_model.predict(
            sequence.reshape(1, HOLISTIC_SEQUENCE_LENGTH, HOLISTIC_NUM_FEATURES),
            verbose=0,
        )
        idx, conf = self._get_prediction(pred)

        # Record metrics
        PREDICTION_LATENCY.labels(model_type="holistic_sequence").observe(
            time.time() - start
        )
        PREDICTIONS_TOTAL.labels(model_type="holistic_sequence", status="success").inc()
        PREDICTION_CONFIDENCE.labels(model_type="holistic_sequence").observe(conf)

        return {
            "word": self.holistic_labels.get(idx, "UNKNOWN"),
            "confidence": round(conf, 2),
            "type": "holistic_medical",
            "processing_time_ms": round((time.time() - start) * 1000, 2),
        }

    def predict_lse(self, landmarks: np.ndarray) -> dict[str, Any] | None:
        """Predict LSE sign from 75 landmarks (MSG3D)."""
        # Shape: (75, 3)
        self._validate_shape(landmarks, (75, 3))
        self.lse_buffer.append(landmarks)

        # Allow prediction with fewer frames? Or wait for buffer fill?
        # MSG3D handles variable length (Pooling).
        # But let's require at least 15 frames to be meaningful.
        if len(self.lse_buffer) < 15:
            return None

        # Pass sequence to MSG3D predictor
        sequence = np.array(list(self.lse_buffer))
        result = self.msg3d_predictor.predict(sequence)

        return {
            "word": result["word"],
            "confidence": result["confidence"],
            "type": "lse_msg3d",
            "processing_time_ms": result["processing_time_ms"],
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

        if buffer_type in ["lse", "all"]:
            self.lse_buffer.clear()

    def get_models_info(self) -> dict[str, Any]:
        return {
            "static": {"count": len(self.static_labels), "type": "alphabet"},
            "dynamic": {"count": len(self.lstm_labels), "type": "alphabet"},
            "words": {"count": len(self.words_labels), "type": "vocabulary"},
            "holistic": {"count": len(self.holistic_labels), "type": "medical"},
            "lse": {"count": len(self.msg3d_predictor.labels), "type": "lse_300"},
        }


# === Singleton ===
_predictor: SignPredictor | None = None


def get_predictor() -> SignPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SignPredictor()
    return _predictor
