"""SignSpeak LSM - Unified Realtime Demo with Holistic Support."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import cv2
import numpy as np
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import (
    HAND_LANDMARKER_PATH,
    SEQUENCE_LENGTH, HOLISTIC_SEQUENCE_LENGTH,
    PREDICTION_INTERVAL
)
from core.predictor import SignPredictor
from core.holistic_extractor import HolisticExtractor


class UnifiedDemo:
    """Unified demo: Alphabet, Words, and Holistic Medical modes."""
    
    GREEN, ORANGE, BLUE, PURPLE = (0,255,0), (0,165,255), (255,150,0), (255,0,150)
    WHITE, BLACK, GRAY = (255,255,255), (0,0,0), (100,100,100)
    
    MODE_ALPHABET, MODE_WORDS, MODE_HOLISTIC = 0, 1, 2
    MODE_NAMES = ["ALPHABET", "WORDS (249)", "HOLISTIC (150)"]
    MODE_COLORS = [(0,255,0), (255,150,0), (255,0,150)]
    
    def __init__(self):
        print("=" * 60)
        print("SIGNSPEAK LSM - UNIFIED DEMO")
        print("=" * 60)
        
        self.predictor = SignPredictor()
        
        # Hand detector for alphabet/words modes
        base_options = python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH))
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        self.hand_detector = vision.HandLandmarker.create_from_options(options)
        
        # Holistic extractor for medical mode
        self.holistic = HolisticExtractor()
        
        self._init_state()
        self._print_controls()
    
    def _init_state(self):
        self.mode = self.MODE_ALPHABET
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.holistic_buffer = deque(maxlen=HOLISTIC_SEQUENCE_LENGTH)
        self.prediction = ""
        self.confidence = 0.0
        self.is_dynamic = False
        self.frame_count = 0
        self.cooldown = 0
    
    def _print_controls(self):
        print("\nControls:")
        print("  1 = Alphabet | 2 = Words | 3 = Holistic")
        print("  C = Clear | Q = Quit")
        print("=" * 60)
    
    def extract_hand_landmarks(self, hand_landmarks) -> np.ndarray:
        wrist = hand_landmarks[0]
        return np.array([
            coord for lm in hand_landmarks 
            for coord in (lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z)
        ])
    
    def detect_movement(self) -> bool:
        if len(self.frame_buffer) < 10:
            return False
        frames = list(self.frame_buffer)
        total = np.linalg.norm(np.array(frames[-1]) - np.array(frames[0]))
        recent = sum(np.linalg.norm(np.array(frames[-i]) - np.array(frames[-i-1])) for i in range(1,5)) / 4
        return total > 0.15 and recent > 0.02
    
    def process_alphabet(self, landmarks: np.ndarray):
        self.frame_buffer.append(landmarks)
        has_movement = self.detect_movement()
        
        if has_movement and len(self.frame_buffer) >= SEQUENCE_LENGTH:
            if self.frame_count % PREDICTION_INTERVAL == 0 and self.cooldown <= 0:
                result = self.predictor.predict_dynamic(np.array(list(self.frame_buffer)))
                if result["confidence"] > 60:
                    self.prediction, self.confidence = result["letter"], result["confidence"]
                    self.is_dynamic, self.cooldown = True, 2
        else:
            self.is_dynamic = False
            if self.frame_count % PREDICTION_INTERVAL == 0:
                result = self.predictor.predict_static(landmarks)
                self.prediction, self.confidence = result["letter"], result["confidence"]
        
        if self.cooldown > 0:
            self.cooldown -= 1
    
    def process_words(self, landmarks: np.ndarray):
        result = self.predictor.predict_word_with_buffer(landmarks)
        if result and result.get("accepted"):
            self.prediction, self.confidence = result["word"], result["confidence"]
        elif len(self.predictor.words_buffer) > 0:
            self.prediction = f"[{len(self.predictor.words_buffer)}/{SEQUENCE_LENGTH}]"
            self.confidence = 0
    
    def process_holistic(self, frame_rgb: np.ndarray):
        features, results = self.holistic.process_frame(frame_rgb)
        
        if features is not None:
            self.holistic_buffer.append(features)
            
            if len(self.holistic_buffer) >= HOLISTIC_SEQUENCE_LENGTH:
                result = self.predictor.predict_holistic(features)
                if result:
                    self.prediction, self.confidence = result["word"], result["confidence"]
            else:
                self.prediction = f"[{len(self.holistic_buffer)}/{HOLISTIC_SEQUENCE_LENGTH}]"
                self.confidence = 0
        
        return results
    
    def draw_ui(self, frame, detected: bool, holistic_results=None):
        h, w = frame.shape[:2]
        color = self.MODE_COLORS[self.mode]
        
        # Header
        cv2.rectangle(frame, (0,0), (w,70), self.BLACK, -1)
        cv2.rectangle(frame, (0,68), (w,70), color, -1)
        cv2.putText(frame, "SignSpeak LSM", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.WHITE, 2)
        cv2.putText(frame, f"Mode: {self.MODE_NAMES[self.mode]}", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Prediction box
        if detected and self.prediction:
            cv2.rectangle(frame, (w-160,10), (w-10,60), color, -1)
            text = self.prediction[:8]
            scale = 1.2 if len(text) <= 3 else 0.7
            cv2.putText(frame, text, (w-150,48), cv2.FONT_HERSHEY_SIMPLEX, scale, self.BLACK, 2)
            if self.confidence > 0:
                cv2.putText(frame, f"{self.confidence:.0f}%", (w-155,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.BLACK, 1)
        
        # Buffer progress
        if self.mode == self.MODE_HOLISTIC:
            buf, mx = len(self.holistic_buffer), HOLISTIC_SEQUENCE_LENGTH
        elif self.mode == self.MODE_WORDS:
            buf, mx = len(self.predictor.words_buffer), SEQUENCE_LENGTH
        else:
            buf, mx = len(self.frame_buffer), SEQUENCE_LENGTH
        
        cv2.rectangle(frame, (10,75), (200,85), (50,50,50), -1)
        cv2.rectangle(frame, (10,75), (int(10 + 190 * buf/mx),85), color, -1)
        cv2.putText(frame, f"Buffer: {buf}/{mx}", (210,83), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.GRAY, 1)
        
        # Phrase (words mode)
        if self.mode == self.MODE_WORDS:
            phrase = self.predictor.get_current_phrase()
            if phrase:
                cv2.rectangle(frame, (10,90), (w-10,120), (30,30,30), -1)
                cv2.putText(frame, f"Frase: {phrase[-50:]}", (15,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1)
        
        # Draw holistic skeleton if available
        if holistic_results:
            self.holistic.draw(frame, holistic_results)
        
        # Footer
        cv2.rectangle(frame, (0,h-30), (w,h), self.BLACK, -1)
        status = "Detected" if detected else "Show hand/body"
        cv2.putText(frame, status, (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.GREEN if detected else self.GRAY, 1)
        cv2.putText(frame, "1/2/3=Mode C=Clear Q=Quit", (w-250,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.GRAY, 1)
    
    def draw_hand(self, frame, hand_landmarks):
        h, w = frame.shape[:2]
        for lm in hand_landmarks:
            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, self.MODE_COLORS[self.mode], -1)
        connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
                       (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
        for s,e in connections:
            cv2.line(frame, (int(hand_landmarks[s].x*w),int(hand_landmarks[s].y*h)),
                     (int(hand_landmarks[e].x*w),int(hand_landmarks[e].y*h)), self.WHITE, 2)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("\nCamera started...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected, holistic_results = False, None
            
            if self.mode == self.MODE_HOLISTIC:
                holistic_results = self.process_holistic(frame_rgb)
                detected = holistic_results is not None and (
                    holistic_results.pose_landmarks or 
                    holistic_results.left_hand_landmarks or 
                    holistic_results.right_hand_landmarks
                )
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = self.hand_detector.detect(mp_image)
                detected = bool(result.hand_landmarks)
                
                if detected:
                    hand = result.hand_landmarks[0]
                    self.draw_hand(frame, hand)
                    landmarks = self.extract_hand_landmarks(hand)
                    
                    if self.mode == self.MODE_ALPHABET:
                        self.process_alphabet(landmarks)
                    else:
                        self.process_words(landmarks)
                    
                    self.frame_count += 1
            
            if not detected and self.mode != self.MODE_HOLISTIC:
                self.reset_state()
            
            self.draw_ui(frame, detected, holistic_results)
            cv2.imshow("SignSpeak LSM", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.switch_mode(self.MODE_ALPHABET)
            elif key == ord('2'):
                self.switch_mode(self.MODE_WORDS)
            elif key == ord('3'):
                self.switch_mode(self.MODE_HOLISTIC)
            elif key == ord('c'):
                self.predictor.reset_buffer("all")
                self.holistic_buffer.clear()
                print("Buffers cleared")
        
        cap.release()
        self.holistic.close()
        cv2.destroyAllWindows()
        print("Demo finished")
    
    def switch_mode(self, mode: int):
        self.mode = mode
        self.reset_state()
        self.holistic_buffer.clear()
        print(f"Mode: {self.MODE_NAMES[mode]}")
    
    def reset_state(self):
        self.frame_buffer.clear()
        self.prediction, self.confidence = "", 0
        self.is_dynamic, self.frame_count, self.cooldown = False, 0, 0


if __name__ == "__main__":
    UnifiedDemo().run()