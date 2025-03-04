import cv2
import numpy as np
import pygame
import tkinter as tk
from tkinter import Button, Label, Canvas, Frame
from PIL import Image, ImageTk
import mediapipe as mp
import tensorflow as tf

# Load the trained AI model
model = tf.keras.models.load_model("drum_ai_model.h5")

# Initialize Pygame for sound playback
pygame.mixer.init()

# Load drum sounds
sound_dir = "sounds/"
drum_sounds = {
    "snare": pygame.mixer.Sound(sound_dir + "snare.wav"),
    "kick": pygame.mixer.Sound(sound_dir + "kick.wav"),
    "hi_hat": pygame.mixer.Sound(sound_dir + "hi_hat.wav"),
    "tom": pygame.mixer.Sound(sound_dir + "tom.wav"),
    "crash": pygame.mixer.Sound(sound_dir + "crash.wav")
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Drum positions based on real drum set layout
drum_positions = {
    "hi_hat": (100, 250),
    "snare": (250, 350),
    "tom": (400, 250),
    "kick": (250, 500),
    "crash": (550, 200)
}

# Create the GUI window
root = tk.Tk()
root.title("AirDrums - Virtual Drum Kit")
root.geometry("1000x750")
root.configure(bg="black")

# Welcome Frame
welcome_frame = Frame(root, bg="black")
welcome_frame.pack(fill="both", expand=True)

welcome_label = Label(welcome_frame, text="Welcome to AirDrums!", font=("Arial", 24), fg="white", bg="black")
welcome_label.pack(pady=20)

instructions_label = Label(welcome_frame, text="Click 'Start Playing' to begin!\nUse your hands to play virtual drums!", font=("Arial", 16), fg="white", bg="black")
instructions_label.pack(pady=10)

start_button = Button(welcome_frame, text="Start Playing", font=("Arial", 16), bg="blue", fg="white", command=lambda: start_application())
start_button.pack(pady=20)

# Main Application Frame
app_frame = Frame(root, bg="black")

drum_canvas = Canvas(app_frame, width=800, height=400, bg="black", highlightthickness=0)
drum_canvas.pack()

drum_img = Image.open("drum_set.png")
drum_img = drum_img.resize((800, 400))  # Resize to fit UI
drum_photo = ImageTk.PhotoImage(drum_img)
drum_canvas.create_image(400, 200, image=drum_photo)

video_label = Label(app_frame, bg="black")
video_label.pack()

# Start function
def start_application():
    welcome_frame.pack_forget()
    app_frame.pack(fill="both", expand=True)
    run_drum_detection()

# Run drum detection with OpenCV
def run_drum_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access camera.")
        return

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
        
        if len(keypoints) == 63:
            keypoints = np.array(keypoints).reshape(1, -1)
            prediction = model.predict(keypoints)
            predicted_class = list(drum_sounds.keys())[np.argmax(prediction)]
            
            # Highlight the detected drum
            if predicted_class in drum_positions:
                drum_x, drum_y = drum_positions[predicted_class]
                cv2.circle(frame, (drum_x, drum_y), 50, (0, 255, 0), -1)
                drum_sounds[predicted_class].play()

        # Convert the OpenCV image to Tkinter format
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, update_frame)

    update_frame()

root.mainloop()