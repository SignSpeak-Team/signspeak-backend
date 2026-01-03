import cv2
import mediapipe as mp
import joblib
import numpy as np
from pathlib import Path


# Ruta al modelo
MODEL_PATH = Path(__file__).parent / "model.pkl"


class HandGestureRecognizer:
    def __init__(self):
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Cargar modelo
        self.model = joblib.load(MODEL_PATH)

    def extract_vector(self, hand_landmarks):
        """Normaliza landmarks y devuelve vector plano de 63 valores"""
        wrist = hand_landmarks.landmark[0]

        normalized = []
        for lm in hand_landmarks.landmark:
            normalized.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])

        return normalized

    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ No se puede abrir la cámara")
            return

        print("📷 Cámara abierta. Presiona 'q' para salir")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    # Dibujar mano
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

                    # Vector de entrada
                    vector = self.extract_vector(hand_landmarks)

                    if len(vector) == 63:
                        prediction = self.model.predict([vector])[0]

                        # Mostrar predicción
                        cv2.putText(
                            frame,
                            f"Gesture: {prediction}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

            cv2.imshow("Vision Service - Hand Gesture", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


