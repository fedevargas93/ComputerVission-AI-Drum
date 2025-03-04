from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PyQt5.QtCore import QTimer, Qt
import cv2
import numpy as np
import pygame
import mediapipe as mp
import tensorflow as tf
import os

class AirDrumsApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AirDrums - PyQt Application")
        self.setGeometry(100, 100, 1000, 750)

        # Load the trained AI model
        self.model = tf.keras.models.load_model("drum_ai_model.h5")

        # Initialize Pygame for sound playback
        pygame.mixer.init()
        
        # Load drum sounds
        sound_dir = "sounds/"
        self.drum_sounds = {
            "snare": pygame.mixer.Sound(os.path.join(sound_dir, "snare.wav")),
            "kick": pygame.mixer.Sound(os.path.join(sound_dir, "kick.wav")),
            "hi_hat": pygame.mixer.Sound(os.path.join(sound_dir, "hi_hat.wav")),
            "tom": pygame.mixer.Sound(os.path.join(sound_dir, "tom.wav")),
            "crash": pygame.mixer.Sound(os.path.join(sound_dir, "crash.wav"))
        }

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
        self.mp_draw = mp.solutions.drawing_utils

        # Adjusted drum positions to reflect drummer's perspective
        self.drum_positions = {
            "hi_hat": (200, 200),  # Hi-hat moved to where crash was
            "snare": (400, 350),
            "tom": (600, 250),
            "kick": (500, 500),  # Centered kick drum
            "crash": (850, 200)   # Crash moved all the way to the right
        }

        # UI Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()

        # Instructions Label
        self.label_instructions = QLabel("🥁 Welcome to AirDrums! 🥁\n\n🔹 How to Use:\n1️⃣ Position your hands in front of the camera\n2️⃣ Hit virtual drums using hand gestures\n3️⃣ Sounds will play based on movement\n4️⃣ Click 'Start Playing' to begin!", self)
        self.label_instructions.setAlignment(Qt.AlignCenter)
        self.label_instructions.setStyleSheet("color: white; font-size: 16px; background-color: black;")
        self.layout.addWidget(self.label_instructions)

        # Start Button
        self.start_button = QPushButton("▶ Start Playing", self)
        self.start_button.clicked.connect(self.start_application)
        self.layout.addWidget(self.start_button)
        
        # Exit Button
        self.exit_button = QPushButton("❌ Exit", self)
        self.exit_button.clicked.connect(self.close_application)
        self.layout.addWidget(self.exit_button)
        
        # Video Label (for camera feed)
        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        self.central_widget.setLayout(self.layout)

        # Camera Setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Adjust camera width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Adjust camera height
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Load and flip the Drum Set Image to match drummer's perspective
        self.drum_set_img = None
        drum_img_path = "drum_set.png"
        if os.path.exists(drum_img_path):
            self.drum_set_img = cv2.imread(drum_img_path)
            self.drum_set_img = cv2.flip(self.drum_set_img, 1)  # Flip the image horizontally to invert view

    def start_application(self):
        self.label_instructions.hide()
        self.start_button.hide()
        self.exit_button.hide()
        self.timer.start(10)

    def close_application(self):
        self.cap.release()
        self.close()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Ensure drum set image matches the camera frame size
        if self.drum_set_img is not None:
            self.drum_set_img = cv2.resize(self.drum_set_img, (w, h))
            frame = cv2.addWeighted(self.drum_set_img, 0.5, frame, 0.5, 0)

        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
        
        if len(keypoints) == 63:
            keypoints = np.array(keypoints).reshape(1, -1)
            prediction = self.model.predict(keypoints)
            predicted_class = list(self.drum_sounds.keys())[np.argmax(prediction)]
            
            # Highlight the detected drum
            if predicted_class in self.drum_positions:
                drum_x, drum_y = self.drum_positions[predicted_class]
                cv2.circle(frame, (drum_x, drum_y), 50, (0, 255, 0), -1)
                self.drum_sounds[predicted_class].play()

        # Overlay drum positions
        for drum, (x, y) in self.drum_positions.items():
            cv2.circle(frame, (x, y), 40, (255, 0, 0), 2)
            cv2.putText(frame, drum.upper(), (x - 30, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Convert OpenCV image to Qt format
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

if __name__ == "__main__":
    app = QApplication([])
    window = AirDrumsApp()
    window.show()
    app.exec_()