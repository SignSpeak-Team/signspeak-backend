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
SIGN_MODEL_PATH = MODELS_DIR / "sign_model.keras"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_letters.keras"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
LSTM_LABEL_ENCODER_PATH = MODELS_DIR / "lstm_label_encoder.pkl"
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
