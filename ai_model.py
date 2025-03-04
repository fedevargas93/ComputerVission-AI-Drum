import os
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands for training
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Define dataset path
data_path = "dataset/"

# Drum classes
classes = ["snare", "kick", "hi_hat", "tom", "crash"]

# Function to check if dataset exists
def check_dataset():
    print("Checking dataset...")
    if not os.path.exists(data_path):
        print("❌ Dataset folder not found!")
        return False
    
    empty_classes = []
    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        if not os.path.exists(cls_path) or len(os.listdir(cls_path)) == 0:
            empty_classes.append(cls)
    
    if empty_classes:
        print(f"⚠️ Warning: No data found for classes: {', '.join(empty_classes)}")
        return False
    
    print("✅ Dataset check complete. All classes have data.")
    return True

# Create AI Model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,)),  # 21 landmarks x 3 (x, y, z)
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(classes), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train AI Model
def train_model():
    if not check_dataset():
        print("❌ Training aborted: Dataset is incomplete.")
        return
    
    print("📊 Loading dataset for training...")
    X, y = [], []
    
    for cls_index, cls in enumerate(classes):
        cls_folder = os.path.join(data_path, cls)
        for file in os.listdir(cls_folder):
            keypoints = np.load(os.path.join(cls_folder, file))
            X.append(keypoints)
            y.append(cls_index)
    
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=len(classes))
    
    model = create_model()
    print("🚀 Starting model training...")
    model.fit(X, y, epochs=20, batch_size=16)
    model.save("drum_ai_model.h5")
    print("✅ Training complete! Model saved as drum_ai_model.h5")

if __name__ == "__main__":
    train_model()