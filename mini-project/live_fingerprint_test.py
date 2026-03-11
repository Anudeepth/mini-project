import time
import serial.tools.list_ports
import tensorflow as tf
import numpy as np
import cv2
from pyfingerprint.pyfingerprint import PyFingerprint

# -----------------------------
# Blood group labels (must match training order)
# -----------------------------
classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# -----------------------------
# Detect fingerprint scanner port automatically
# -----------------------------
ports = [port.device for port in serial.tools.list_ports.comports()]
port = next((p for p in ports if 'USB' in p.upper() or 'ACM' in p.upper()), None)

if not port:
    print("❌ Scanner not detected")
    exit()

print("✅ Scanner detected on:", port)

try:
    # -----------------------------
    # Connect to fingerprint sensor
    # -----------------------------
    f = PyFingerprint(port, 57600, 0xFFFFFFFF, 0x00000000)
    if not f.verifyPassword():
        print("❌ Scanner password error")
        exit()

    print("Place your finger on the scanner...")

    while not f.readImage():
        pass

    print("✅ Fingerprint captured")

    # -----------------------------
    # Save fingerprint image
    # -----------------------------
    img_path = "/home/anudeepth/Documents/fingerprint_raw.bmp"
    f.downloadImage(img_path)
    print("✅ Image saved:", img_path)

    # -----------------------------
    # Load trained model
    # -----------------------------
    print("Loading model...")
    model = tf.keras.models.load_model("fingerprint_model.keras")

    # -----------------------------
    # Preprocessing function
    # -----------------------------
    def preprocess_fingerprint(img_path):
        img = cv2.imread(img_path)  # Load in color (3 channels)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # CLAHE on each channel for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        channels = cv2.split(img)
        channels = [clahe.apply(ch) for ch in channels]
        img = cv2.merge(channels)

        # Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (3,3), 0)

        # Morphological gradient to enhance ridges
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

        # Resize to model input
        gray = cv2.resize(gray, (128,128))

        # Normalize
        gray = gray / 255.0

        # Expand dims to match model input and repeat to 3 channels
        gray = np.expand_dims(gray, axis=0)           # (1,128,128)
        gray = np.repeat(gray[..., np.newaxis], 3, axis=-1)  # (1,128,128,3)
        return gray

    # -----------------------------
    # Prediction
    # -----------------------------
    img = preprocess_fingerprint(img_path)
    prediction = model.predict(img)
    index = np.argmax(prediction)
    blood = classes[index]
    confidence = prediction[0][index] * 100

    print("\n===== RESULT =====")
    print("Predicted Blood Group:", blood)
    print("Confidence: {:.2f}%".format(confidence))

except Exception as e:
    print("❌ Error:", e)