"""
SignSpeak LSM - Unified Realtime Demo
All 4 models: Static (21), Dynamic (6), Words (249), Holistic Medical (150)
"""

import sys
import os

# Add src to path for imports
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


class UnifiedDemo:
    """Unified demo with all models and keyboard mode switching."""
    
    # Colors
    GREEN = (0, 255, 0)
    ORANGE = (0, 165, 255)
    BLUE = (255, 150, 0)
    PURPLE = (255, 0, 150)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (100, 100, 100)
    
    # Modes
    MODE_ALPHABET = 0    # Static + Dynamic letters
    MODE_WORDS = 1       # 249 words
    MODE_HOLISTIC = 2    # 150 medical (placeholder - needs holistic landmarks)
    
    MODE_NAMES = ["ALPHABET", "WORDS (249)", "HOLISTIC MEDICAL"]
    MODE_COLORS = [GREEN, BLUE, PURPLE]
    
    def __init__(self):
        print("=" * 60)
        print("SIGNSPEAK LSM - UNIFIED DEMO")
        print("=" * 60)
        
        # Load predictor with all models
        self.predictor = SignPredictor()
        
        # Setup MediaPipe
        base_options = python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH))
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # State
        self.mode = self.MODE_ALPHABET
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.current_type = ""
        self.is_dynamic = False
        self.frame_count = 0
        self.cooldown = 0
        
        print("\nControls:")
        print("  1 = Alphabet mode (static + dynamic letters)")
        print("  2 = Words mode (249 vocabulary)")
        print("  3 = Holistic mode (150 medical - requires holistic landmarks)")
        print("  C = Clear phrase buffer")
        print("  Q = Quit")
        print("=" * 60)
    
    def extract_landmarks(self, hand_landmarks) -> np.ndarray:
        """Extract normalized 63-feature vector from hand landmarks."""
        wrist = hand_landmarks[0]
        vector = []
        for lm in hand_landmarks:
            vector.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        return np.array(vector)
    
    def detect_movement(self) -> bool:
        """Detect significant hand movement in buffer."""
        if len(self.frame_buffer) < 10:
            return False
        
        frames = list(self.frame_buffer)
        first, last = np.array(frames[0]), np.array(frames[-1])
        total_movement = np.linalg.norm(last - first)
        
        # Check recent movement (last 5 frames)
        recent = frames[-5:]
        if len(recent) >= 5:
            recent_movement = sum(
                np.linalg.norm(np.array(recent[i+1]) - np.array(recent[i]))
                for i in range(4)
            ) / 4
            if recent_movement < 0.02:
                return False
        
        return total_movement > 0.15
    
    def draw_landmarks(self, image, hand_landmarks):
        """Draw hand skeleton on image."""
        h, w = image.shape[:2]
        
        # Draw points
        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (x, y), 4, self.MODE_COLORS[self.mode], -1)
        
        # Draw connections
        connections = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),(0,17)
        ]
        for start, end in connections:
            x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
            x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
            cv2.line(image, (x1,y1), (x2,y2), self.WHITE, 2)
    
    def process_alphabet_mode(self, landmarks: np.ndarray):
        """Process alphabet prediction (static + dynamic)."""
        self.frame_buffer.append(landmarks)
        has_movement = self.detect_movement()
        
        if has_movement and len(self.frame_buffer) >= SEQUENCE_LENGTH:
            if self.frame_count % PREDICTION_INTERVAL == 0 and self.cooldown <= 0:
                sequence = np.array(list(self.frame_buffer))
                result = self.predictor.predict_dynamic(sequence)
                if result["confidence"] > 60:
                    self.current_prediction = result["letter"]
                    self.current_confidence = result["confidence"]
                    self.current_type = "dynamic"
                    self.is_dynamic = True
                    self.cooldown = 2
        else:
            self.is_dynamic = False
            if self.frame_count % PREDICTION_INTERVAL == 0:
                result = self.predictor.predict_static(landmarks)
                self.current_prediction = result["letter"]
                self.current_confidence = result["confidence"]
                self.current_type = "static"
        
        if self.cooldown > 0:
            self.cooldown -= 1
    
    def process_words_mode(self, landmarks: np.ndarray):
        """Process words prediction with buffer."""
        result = self.predictor.predict_word_with_buffer(landmarks)
        
        if result and result.get("accepted"):
            self.current_prediction = result["word"]
            self.current_confidence = result["confidence"]
            self.current_type = "word"
        elif self.predictor.words_buffer:
            # Show buffering progress
            self.current_prediction = f"[{len(self.predictor.words_buffer)}/{SEQUENCE_LENGTH}]"
            self.current_confidence = 0
            self.current_type = "buffering"
    
    def draw_ui(self, frame, hand_detected: bool):
        """Draw UI overlay."""
        h, w = frame.shape[:2]
        mode_color = self.MODE_COLORS[self.mode]
        
        # Header
        cv2.rectangle(frame, (0, 0), (w, 70), self.BLACK, -1)
        cv2.rectangle(frame, (0, 68), (w, 70), mode_color, -1)
        
        cv2.putText(frame, "SignSpeak LSM", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.WHITE, 2)
        cv2.putText(frame, f"Mode: {self.MODE_NAMES[self.mode]}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        
        # Prediction box
        if hand_detected and self.current_prediction:
            cv2.rectangle(frame, (w-160, 10), (w-10, 60), mode_color, -1)
            
            # Truncate long words
            display_text = self.current_prediction[:8]
            font_scale = 1.2 if len(display_text) <= 3 else 0.7
            
            cv2.putText(frame, display_text, (w-150, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.BLACK, 2)
            if self.current_confidence > 0:
                cv2.putText(frame, f"{self.current_confidence:.0f}%", (w-155, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.BLACK, 1)
        else:
            cv2.rectangle(frame, (w-160, 10), (w-10, 60), (50, 50, 50), -1)
            cv2.putText(frame, "---", (w-100, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.GRAY, 2)
        
        # Buffer progress (for dynamic/words modes)
        if self.mode != self.MODE_HOLISTIC:
            buffer_len = len(self.frame_buffer) if self.mode == self.MODE_ALPHABET else len(self.predictor.words_buffer)
            max_len = SEQUENCE_LENGTH
            progress = buffer_len / max_len
            
            cv2.rectangle(frame, (10, 75), (200, 85), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 75), (int(10 + 190 * progress), 85), mode_color, -1)
            cv2.putText(frame, f"Buffer: {buffer_len}/{max_len}", (210, 83),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.GRAY, 1)
        
        # Phrase display (words mode only)
        if self.mode == self.MODE_WORDS:
            phrase = self.predictor.get_current_phrase()
            if phrase:
                cv2.rectangle(frame, (10, 90), (w-10, 120), (30, 30, 30), -1)
                cv2.putText(frame, f"Phrase: {phrase[-50:]}", (15, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1)
        
        # Footer
        cv2.rectangle(frame, (0, h-30), (w, h), self.BLACK, -1)
        status = "Hand detected" if hand_detected else "Show your hand"
        cv2.putText(frame, status, (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    self.GREEN if hand_detected else self.GRAY, 1)
        cv2.putText(frame, "1/2/3=Mode C=Clear Q=Quit", (w-250, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.GRAY, 1)
    
    def run(self):
        """Main loop."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\nCamera started...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            result = self.detector.detect(mp_image)
            hand_detected = bool(result.hand_landmarks)
            
            if hand_detected:
                hand_landmarks = result.hand_landmarks[0]
                self.draw_landmarks(frame, hand_landmarks)
                
                landmarks = self.extract_landmarks(hand_landmarks)
                
                if self.mode == self.MODE_ALPHABET:
                    self.process_alphabet_mode(landmarks)
                elif self.mode == self.MODE_WORDS:
                    self.process_words_mode(landmarks)
                elif self.mode == self.MODE_HOLISTIC:
                    # Holistic requires MediaPipe Holistic (pose + hands)
                    # Currently placeholder - only hand landmarks available
                    self.current_prediction = "Need Holistic"
                    self.current_confidence = 0
                
                self.frame_count += 1
            else:
                self.reset_state()
            
            self.draw_ui(frame, hand_detected)
            cv2.imshow("SignSpeak LSM - Unified Demo", frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.mode = self.MODE_ALPHABET
                self.reset_state()
                print("Mode: ALPHABET")
            elif key == ord('2'):
                self.mode = self.MODE_WORDS
                self.reset_state()
                print("Mode: WORDS")
            elif key == ord('3'):
                self.mode = self.MODE_HOLISTIC
                self.reset_state()
                print("Mode: HOLISTIC (requires holistic landmarks)")
            elif key == ord('c'):
                self.predictor.reset_buffer("all")
                print("Buffers cleared")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Demo finished")
    
    def reset_state(self):
        """Reset prediction state."""
        self.frame_buffer.clear()
        self.current_prediction = ""
        self.current_confidence = 0
        self.is_dynamic = False
        self.frame_count = 0
        self.cooldown = 0


if __name__ == "__main__":
    demo = UnifiedDemo()
    demo.run()