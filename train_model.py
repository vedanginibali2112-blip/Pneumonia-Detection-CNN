import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning messages

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import numpy as np
from sklearn.utils import shuffle

IMG_SIZE = 150  # Size to which each image will be resized
DATA_DIR = 'data'  # Path to folder containing 'PNEUMONIA' and 'NORMAL' subfolders

# Function to load and preprocess data from folders
def load_data(data_dir):
    images = []
    labels = []
    categories = ["PNEUMONIA", "NORMAL"]
    
    for label in categories:
        path = os.path.join(data_dir, label)
        if not os.path.exists(path):
            print(f"Warning: Directory '{path}' not found.")
            continue
        
        class_num = 1 if label == "PNEUMONIA" else 0
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    continue  # Skip if image is unreadable
                resized_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                images.append(resized_img)
                labels.append(class_num)
            except Exception as e:
                print(f"Failed to process image {img_name}: {e}")
    
    return np.array(images), np.array(labels)

# Define and compile a simple CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess data
images, labels = load_data(DATA_DIR)

# Check for empty dataset
if images.size == 0 or labels.size == 0:
    raise ValueError("No images found. Please check your dataset structure.")

# Shuffle and normalize the data
images, labels = shuffle(images, labels, random_state=42)
images = images.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0

# Initialize model and train
model = create_model()
model.fit(images, labels, epochs=10, validation_split=0.2)

# Save the trained model
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/pneumonia_detection_model.h5")
print("✅ Model training complete and saved in 'saved_model' folder.")

