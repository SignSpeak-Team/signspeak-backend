"""Vision Service Configuration."""

from pathlib import Path

# === Paths ===
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# === Model Paths ===
# Alphabet - Static (21 letters)
SIGN_MODEL_PATH = MODELS_DIR / "sign_model.keras"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# Alphabet - Dynamic LSTM (J, K, Q, X, Z, Ñ)
LSTM_MODEL_PATH = MODELS_DIR / "lstm_letters.keras"
LSTM_LABEL_ENCODER_PATH = MODELS_DIR / "lstm_label_encoder.pkl"

# Words - 249 general vocabulary
WORDS_MODEL_PATH = MODELS_DIR / "words_model.keras"
WORDS_LABEL_ENCODER_PATH = MODELS_DIR / "words_label_encoder.pkl"

# Holistic - 150 medical vocabulary (pose + hands)
HOLISTIC_MODEL_PATH = MODELS_DIR / "best_model.h5"
HOLISTIC_LABEL_ENCODER_PATH = MODELS_DIR / "holistic_label_encoder.pkl"

# MediaPipe
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"

# === Model Input Shapes ===
SEQUENCE_LENGTH = 15           # Frames for alphabet/words
NUM_FEATURES = 63              # 21 hand landmarks * 3

HOLISTIC_SEQUENCE_LENGTH = 30  # Frames for holistic
HOLISTIC_NUM_FEATURES = 226    # Pose + both hands

# === Detection Thresholds ===
MOVEMENT_THRESHOLD = 0.15
RECENT_MOVEMENT_THRESHOLD = 0.02
RECENT_FRAMES_COUNT = 5

# === Confidence ===
HIGH_CONFIDENCE_THRESHOLD = 80
MEDIUM_CONFIDENCE_THRESHOLD = 60

# === Cooldowns (frames) ===
HIGH_CONFIDENCE_COOLDOWN = 2
MEDIUM_CONFIDENCE_COOLDOWN = 1
PREDICTION_INTERVAL = 3
