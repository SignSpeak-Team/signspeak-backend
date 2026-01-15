import numpy as np
import pickle
from tensorflow import keras
from collections import deque
from typing import Dict, Tuple, Optional
from config import (
    SIGN_MODEL_PATH, LSTM_MODEL_PATH,
    LABEL_ENCODER_PATH, LSTM_LABEL_ENCODER_PATH,
    WORDS_MODEL_PATH, WORDS_LABEL_ENCODER_PATH,
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
        
        # Cargar modelo dinámico LSTM (letras)
        self.lstm_model = keras.models.load_model(str(LSTM_MODEL_PATH))
        with open(LSTM_LABEL_ENCODER_PATH, "rb") as f:
            lstm_labels = pickle.load(f)
        self.lstm_idx_to_letter = {v: k for k, v in lstm_labels.items()}
        
        # Cargar modelo LSTM (249 palabras)
        self.words_model = keras.models.load_model(str(WORDS_MODEL_PATH))
        with open(WORDS_LABEL_ENCODER_PATH, "rb") as f:
            words_labels = pickle.load(f)
        self.words_idx_to_word = {v: k for k, v in words_labels.items()}
        
        # Buffers para secuencias
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)  # Para letras dinámicas
        self.words_buffer = deque(maxlen=SEQUENCE_LENGTH)  # Para palabras
        
        print(f"[Predictor] ✓ Modelo estático cargado: {len(self.static_idx_to_letter)} letras")
        print(f"[Predictor] ✓ Modelo LSTM letras cargado: {len(self.lstm_idx_to_letter)} letras")
        print(f"[Predictor] ✓ Modelo LSTM palabras cargado: {len(self.words_idx_to_word)} palabras")
    
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
        
        elapsed_time = time.time() - start_time
        
        return {
            "letter": letter,
            "confidence": confidence,
            "type": "dynamic",
            "time": elapsed_time * 1000
        }
    
    def predict_word(self, landmarks: np.ndarray) -> Dict[str, any]:
        """
        Predice una palabra desde una secuencia de landmarks.
        
        Args:
            landmarks: Array numpy de shape (63,) con coordenadas landmarks del frame actual
        
        Returns:
            dict con 'word', 'confidence', 'type' o None si no hay suficientes frames
        """
        import time
        start_time = time.time()
        
        # Validar input
        if landmarks.shape != (63,):
            raise ValueError(f"Expected shape (63,), got {landmarks.shape}")
        
        # Añadir frame al buffer
        self.words_buffer.append(landmarks)
        
        # Necesitamos SEQUENCE_LENGTH frames
        if len(self.words_buffer) < SEQUENCE_LENGTH:
            return None
        
        # Crear secuencia
        sequence = np.array(list(self.words_buffer))  # Shape: (SEQUENCE_LENGTH, 63)
        sequence_reshaped = np.array([sequence])  # Shape: (1, SEQUENCE_LENGTH, 63)
        
        # Predicción
        prediction = self.words_model.predict(sequence_reshaped, verbose=0)
        
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx] * 100)
        word = self.words_idx_to_word.get(class_idx, "UNKNOWN")
        
        elapsed_time = time.time() - start_time
        
        return {
            "word": word,
            "confidence": confidence,
            "type": "word",
            "time": elapsed_time * 1000,
            "buffer_size": len(self.words_buffer)
        }
    
    def reset_buffer(self, buffer_type: str = "all"):
        """
        Reinicia los buffers de secuencias.
        
        Args:
            buffer_type: "letters", "words", o "all"
        """
        if buffer_type in ["letters", "all"]:
            self.frame_buffer.clear()
        if buffer_type in ["words", "all"]:
            self.words_buffer.clear()
    
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
            },
            "words_model": {
                "vocabulary_size": len(self.words_idx_to_word),
                "accuracy": 91.24,
                "version": "1.0",
                "categories": "Saludos, Tiempo, Escuela, Familia, Hogar, Personas, Cocina, Ropa, Partes del Cuerpo, Vehículos, Lugares, Pronombres, Verbos, Profesiones, Estados de México"
            }
        }


# Instancia global del predictor (se carga una vez al iniciar la API)
_predictor_instance: Optional[SignPredictor] = None


def get_predictor() -> SignPredictor:

    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = SignPredictor()
    
    return _predictor_instance
