from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ===========================================
# Flask Configuration
# ===========================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = os.path.join('saved_model', 'pneumonia_detection_model.h5')
IMG_SIZE = 150

# ===========================================
# Load Model
# ===========================================
try:
    model = load_model(MODEL_PATH)
    print(f"[INFO] ✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"[ERROR] ❌ Could not load model: {e}")

# ===========================================
# Global Variables
# ===========================================
results = []
total = 0
normal = 0
pneumonia = 0

# ===========================================
# Prediction Function
# ===========================================
def predict_image(img_path):
    try:
        if model is None:
            raise RuntimeError("Model not loaded")

        # Read image in GRAYSCALE since model was trained on grayscale images
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)  # add channel dimension
        img = np.expand_dims(img, axis=0)   # add batch dimension

        # Predict
        pred = model.predict(img)
        prob = float(pred[0][0])
        label = "Pneumonia" if prob >= 0.5 else "Normal"
        confidence = round(prob * 100 if label == "Pneumonia" else (1 - prob) * 100, 2)

        print(f"[PREDICT] {os.path.basename(img_path)} → {label} ({confidence}%)")
        return label, confidence

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return "Error", 0

# ===========================================
# Routes
# ===========================================
@app.route('/')
def index():
    return render_template(
        'dashboard.html',
        results=results,
        total=total,
        normal=normal,
        pneumonia=pneumonia
    )

# ---------- Upload Route ----------
@app.route('/upload', methods=['POST'])
def upload():
    global total, normal, pneumonia

    file = request.files.get('file')
    if not file or not file.filename:
        return redirect(url_for('index'))

    # Retrieve patient information
    patient_id = request.form.get('patient_id')
    patient_name = request.form.get('patient_name')
    patient_age = request.form.get('patient_age')
    patient_gender = request.form.get('patient_gender')

    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Predict
    label, confidence = predict_image(filepath)

    # Update counts
    total += 1
    if label == "Normal":
        normal += 1
    elif label == "Pneumonia":
        pneumonia += 1

    # Store result entry
    image_url = url_for('static', filename=f'uploads/{file.filename}')
    results.insert(0, {
        "id": patient_id,
        "name": patient_name,
        "age": patient_age,
        "gender": patient_gender,
        "image": image_url,
        "result": label,
        "confidence": confidence
    })

    print(f"[INFO] {patient_name} ({patient_age}/{patient_gender}) → {label} ({confidence}%)")
    return redirect(url_for('index'))

# ---------- Clear Dashboard ----------
@app.route('/clear')
def clear():
    global results, total, normal, pneumonia
    results.clear()
    total = normal = pneumonia = 0
    print("[CLEAR] Dashboard data reset.")
    return redirect(url_for('index'))

# ===========================================
# Run Flask App
# ===========================================
if __name__ == '__main__':
    app.run(debug=True)
