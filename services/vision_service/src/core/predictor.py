import numpy as np
import pickle
from tensorflow import keras
from collections import deque
from typing import Dict, Tuple, Optional
from config import (
    SIGN_MODEL_PATH, LSTM_MODEL_PATH,
    LABEL_ENCODER_PATH, LSTM_LABEL_ENCODER_PATH,
    SEQUENCE_LENGTH, HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD
)


class SignPredictor:
    """Predictor de lenguaje de señas (estático y dinámico)"""
    
    def __init__(self):
        """Inicializa modelos y configuración"""
        print("[Predictor] Cargando modelos...")
        
        # Cargar modelo estático
        self.static_model = keras.models.load_model(str(SIGN_MODEL_PATH))
        with open(LABEL_ENCODER_PATH, "rb") as f:
            static_labels = pickle.load(f)
        self.static_idx_to_letter = {v: k for k, v in static_labels.items()}
        
        # Cargar modelo dinámico LSTM
        self.lstm_model = keras.models.load_model(str(LSTM_MODEL_PATH))
        with open(LSTM_LABEL_ENCODER_PATH, "rb") as f:
            lstm_labels = pickle.load(f)
        self.lstm_idx_to_letter = {v: k for k, v in lstm_labels.items()}
        
        # Buffer para secuencias
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
        print(f"[Predictor] ✓ Modelo estático cargado: {len(self.static_idx_to_letter)} letras")
        print(f"[Predictor] ✓ Modelo LSTM cargado: {len(self.lstm_idx_to_letter)} letras")
    
    def predict_static(self, landmarks: np.ndarray) -> Dict[str, any]:
        """
        Predice una letra estática desde landmarks de mano.
        
        Args:
            landmarks: Array numpy de shape (63,) con coordenadas landmarks
        
        Returns:
            dict con 'letter', 'confidence', 'type'
        """
        import time
        start_time = time.time()
        
        # Validar input
        if landmarks.shape != (63,):
            raise ValueError(f"Expected shape (63,), got {landmarks.shape}")
        
        # Predicción
        landmarks_reshaped = np.array([landmarks])
        prediction = self.static_model.predict(landmarks_reshaped, verbose=0)
        
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx] * 100)
        letter = self.static_idx_to_letter[class_idx]
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        return {
            "letter": letter,
            "confidence": round(confidence, 2),
            "type": "static",
            "processing_time_ms": round(processing_time, 2)
        }
    
    def predict_dynamic(self, sequence: np.ndarray) -> Dict[str, any]:
        import time
        start_time = time.time()
        
        # Validar input
        if sequence.shape != (SEQUENCE_LENGTH, 63):
            raise ValueError(f"Expected shape ({SEQUENCE_LENGTH}, 63), got {sequence.shape}")
        
        # Predicción
        sequence_reshaped = np.array([sequence])
        prediction = self.lstm_model.predict(sequence_reshaped, verbose=0)
        
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx] * 100)
        letter = self.lstm_idx_to_letter[class_idx]
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        return {
            "letter": letter,
            "confidence": round(confidence, 2),
            "type": "dynamic",
            "processing_time_ms": round(processing_time, 2)
        }
    
    def get_models_info(self) -> Dict[str, any]:
        return {
            "static_model": {
                "letters": list(self.static_idx_to_letter.values()),
                "count": len(self.static_idx_to_letter),
                "accuracy": 95.2,  # Aproximado
                "version": "2.0"
            },
            "dynamic_model": {
                "letters": list(self.lstm_idx_to_letter.values()),
                "count": len(self.lstm_idx_to_letter),
                "accuracy": 99.79,
                "version": "2.0"
            }
        }


# Instancia global del predictor (se carga una vez al iniciar la API)
_predictor_instance: Optional[SignPredictor] = None


def get_predictor() -> SignPredictor:
    """
    Dependency injection para FastAPI.
    Retorna la instancia del predictor (singleton pattern).
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = SignPredictor()
    
    return _predictor_instance
