"""
Configuration file for SignSpeak Vision Service.
Centralizes all paths and constants.
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent  # vision_service/
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

# Model paths
# Static sign model (21 letters)
SIGN_MODEL_PATH = MODELS_DIR / "sign_model.keras"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# LSTM model (dynamic letters: J, K, Q, X, Z, Ñ)
LSTM_MODEL_PATH = MODELS_DIR / "lstm_letters.keras"
LSTM_LABEL_ENCODER_PATH = MODELS_DIR / "lstm_label_encoder.pkl"

# LSTM model (249 words)
WORDS_MODEL_PATH = MODELS_DIR / "words_model.keras"
WORDS_LABEL_ENCODER_PATH = MODELS_DIR / "words_label_encoder.pkl"

# MediaPipe hand landmarker
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"

# Model configuration
SEQUENCE_LENGTH = 15
NUM_FEATURES = 63
PREDICTION_INTERVAL = 3

# Detection configuration
MOVEMENT_THRESHOLD = 0.15
RECENT_MOVEMENT_THRESHOLD = 0.02
RECENT_FRAMES_COUNT = 5

# Cooldown configuration (frames)
HIGH_CONFIDENCE_COOLDOWN = 2
MEDIUM_CONFIDENCE_COOLDOWN = 1

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 80
MEDIUM_CONFIDENCE_THRESHOLD = 60
