import os
import numpy as np
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands (instead of Holistic)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Define path for saving training data
data_path = "dataset/"
os.makedirs(data_path, exist_ok=True)

# Drum classes
classes = ["snare", "kick", "hi_hat", "tom", "crash"]

# Create folders for each class
for cls in classes:
    os.makedirs(os.path.join(data_path, cls), exist_ok=True)

def capture_training_data(class_name, num_samples=100):
    """
    Captures hand movement data for a specific drum class.
    :param class_name: Name of the class (snare, kick, etc.)
    :param num_samples: Number of samples to collect.
    """
    cap = cv2.VideoCapture(0)
    count = 0
    
    print(f"Capturing data for: {class_name}")
    
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
        
        if len(keypoints) == 63:  # Each hand has 21 landmarks (x, y, z)
            np.save(os.path.join(data_path, class_name, f"sample_{count}.npy"), np.array(keypoints))
            count += 1
            print(f"{count}/{num_samples} samples collected for {class_name}")
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow("Capturing Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection for {class_name} completed.")

if __name__ == "__main__":
    class_name = input("Enter the drum class to capture (snare, kick, hi_hat, tom, crash): ")
    if class_name in classes:
        num_samples = int(input("Number of samples to capture: "))
        capture_training_data(class_name, num_samples)
    else:
        print("Invalid class. Use one of: snare, kick, hi_hat, tom, crash.")
