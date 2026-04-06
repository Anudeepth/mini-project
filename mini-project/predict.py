import json
import time
import serial.tools.list_ports
import tensorflow as tf
import numpy as np
import cv2
from pyfingerprint.pyfingerprint import PyFingerprint

# ─────────────────────────────────────────
# Load class names (saved during training)
# ─────────────────────────────────────────
try:
    with open("class_names.json") as f:
        classes = json.load(f)
except FileNotFoundError:
    # Fallback — alphabetical order matches TensorFlow's folder sorting
    classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

print("Blood group classes:", classes)

# ─────────────────────────────────────────
# Detect scanner port
# ─────────────────────────────────────────
ports = [p.device for p in serial.tools.list_ports.comports()]
port  = next((p for p in ports if "USB" in p.upper() or "ACM" in p.upper()), None)

if port is None:
    print("❌ Fingerprint scanner not detected")
    exit()

print("✅ Scanner detected on:", port)

try:
    # ─────────────────────────────────────
    # Connect to scanner
    # ─────────────────────────────────────
    f = PyFingerprint(port, 57600, 0xFFFFFFFF, 0x00000000)
    if not f.verifyPassword():
        raise ValueError("Scanner password incorrect")
    print("✅ Scanner initialised")

    # ─────────────────────────────────────
    # Capture fingerprint image
    # ─────────────────────────────────────
    print("Place your finger on the scanner...")
    while not f.readImage():
        time.sleep(0.2)
    print("✅ Fingerprint captured")

    img_path = "/home/anudeepth/Documents/mini-project/mini-project/fingerprint_raw.bmp"
    f.downloadImage(img_path)
    print("✅ Image saved:", img_path)

    # ─────────────────────────────────────
    # Load model — ✅ Fixed: use ResNet model
    # ─────────────────────────────────────
    print("Loading model...")
    model = tf.keras.models.load_model("resnet_fingerprint_model.keras")
    print("✅ Model loaded | Input shape:", model.input_shape)

    # ─────────────────────────────────────
    # Preprocessing — ✅ matches training exactly
    # ResNet50V2 expects (128,128,3) in range [-1,1]
    # ─────────────────────────────────────
    def preprocess_fingerprint(path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Fingerprint image could not be loaded: " + path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # --- Smart Scanner ROI CROP ---
        if img_rgb.shape[0] > 200:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                pad = 5
                x, y = max(0, x - pad), max(0, y - pad)
                w = min(img_rgb.shape[1] - x, w + 2*pad)
                h = min(img_rgb.shape[0] - y, h + 2*pad)
                img_rgb = img_rgb[y:y+h, x:x+w]
                
        img = cv2.resize(img_rgb, (128, 128))
        img = img.astype("float32")
        img = np.expand_dims(img, axis=0)
        return img

    img = preprocess_fingerprint(img_path)

    # ─────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────
    prediction = model.predict(img)
    index      = np.argmax(prediction)
    blood      = classes[index]
    confidence = prediction[0][index] * 100

    print("\n========== RESULT ==========")
    print("Predicted Blood Group:", blood)
    print("Confidence: {:.2f}%".format(confidence))
    print()
    print("All class probabilities:")
    for cls, prob in zip(classes, prediction[0]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:4s}: {prob*100:5.1f}%  {bar}")

except Exception as e:
    print("❌ Error:", e)