import json
import sys
import tensorflow as tf
import numpy as np
import cv2

# ─────────────────────────────────────────
# Load class names
# ─────────────────────────────────────────
try:
    with open("class_names.json") as f:
        classes = json.load(f)
except FileNotFoundError:
    classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# ─────────────────────────────────────────
# Load model
# ─────────────────────────────────────────
model = tf.keras.models.load_model("resnet_fingerprint_model.keras")
print("Model input shape:", model.input_shape)

# ─────────────────────────────────────────
# Preprocessing — matches training exactly
# ─────────────────────────────────────────
def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet_v2.preprocess_input(img)
    return img

# Accept image path as argument, default to fingerprint_raw.bmp
img_path = sys.argv[1] if len(sys.argv) > 1 else "fingerprint_raw.bmp"
img = preprocess(img_path)

prediction = model.predict(img)
index = np.argmax(prediction)

print(f"\nPredicted Blood Group: {classes[index]}")
print(f"Confidence: {prediction[0][index]*100:.2f}%")
print("\nAll probabilities:")
for cls, prob in zip(classes, prediction[0]):
    print(f"  {cls:4s}: {prob*100:5.1f}%")