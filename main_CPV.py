import time
import os
import numpy as np
import pygame
import cv2
import mediapipe as mp
import tensorflow as tf

# Cargar modelo entrenado
model = tf.keras.models.load_model("drum_ai_model.h5")

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Inicializar Pygame para reproducir sonidos
pygame.mixer.init()

# Definir rutas de sonido
sound_dir = "sounds/"
drum_sounds = {
    "snare": pygame.mixer.Sound(os.path.join(sound_dir, "snare.wav")),
    "kick": pygame.mixer.Sound(os.path.join(sound_dir, "kick.wav")),
    "hi_hat": pygame.mixer.Sound(os.path.join(sound_dir, "hi_hat.wav")),
    "tom": pygame.mixer.Sound(os.path.join(sound_dir, "tom.wav")),
    "crash": pygame.mixer.Sound(os.path.join(sound_dir, "crash.wav"))
}

# Configurar la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el cuadro de la cámara.")
        break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Convertir a RGB
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
        print(f"Golpe detectado: {predicted_class}")
        
        # Reproducir sonido correspondiente
        if predicted_class in drum_sounds:
            drum_sounds[predicted_class].play()
    
    cv2.imshow("Virtual Drums", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
