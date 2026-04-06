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
    classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

print("Blood group classes:", classes)

# ─────────────────────────────────────────
# Detect scanner port
# ─────────────────────────────────────────
ports = [p.device for p in serial.tools.list_ports.comports()]
port  = next((p for p in ports if "USB" in p.upper() or "ACM" in p.upper()), None)

if not port:
    print("❌ Scanner not detected")
    exit()

print("✅ Scanner detected on:", port)

try:
    # ─────────────────────────────────────
    # Connect to fingerprint sensor
    # ─────────────────────────────────────
    f = PyFingerprint(port, 57600, 0xFFFFFFFF, 0x00000000)
    if not f.verifyPassword():
        print("❌ Scanner password error")
        exit()

    # ─────────────────────────────────────
    # Multi-capture averaging for image quality
    # ─────────────────────────────────────
    NUM_CAPTURES  = 5
    CAPTURE_DELAY = 0.4

    print(f"Place your finger firmly on the scanner...")
    print(f"⏳ Will take {NUM_CAPTURES} scans and average them for better quality...")

    img_path  = "/home/anudeepth/Documents/fingerprint_raw.bmp"
    tmp_paths = []

    for i in range(NUM_CAPTURES):
        print(f"  📷 Scan {i+1}/{NUM_CAPTURES}... Keep your finger still")
        while not f.readImage():
            pass
        tmp_path = f"/home/anudeepth/Documents/fingerprint_tmp_{i}.bmp"
        f.downloadImage(tmp_path)
        tmp_paths.append(tmp_path)
        print(f"  ✅ Scan {i+1} captured")
        if i < NUM_CAPTURES - 1:
            time.sleep(CAPTURE_DELAY)

    print("✅ All scans captured — averaging for best quality...")
    frames  = [cv2.imread(p).astype(np.float32) for p in tmp_paths]
    averaged = np.mean(frames, axis=0).astype(np.uint8)
    cv2.imwrite(img_path, averaged)

    import os
    for p in tmp_paths:
        try:
            os.remove(p)
        except Exception:
            pass

    print("✅ Averaged image saved:", img_path)

    # ─────────────────────────────────────
    # Load model — ✅ Fixed: use ResNet model
    # ─────────────────────────────────────
    print("Loading model...")
    model = tf.keras.models.load_model("resnet_fingerprint_model.keras")
    print("✅ Model loaded | Input shape:", model.input_shape)

    # ─────────────────────────────────────
    # Preprocessing — ✅ matches training exactly
    # ─────────────────────────────────────
    def preprocess_fingerprint(path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Could not load image: " + path)
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

    img        = preprocess_fingerprint(img_path)
    prediction = model.predict(img)
    index      = np.argmax(prediction)
    blood      = classes[index]
    confidence = prediction[0][index] * 100

    print("\n===== RESULT =====")
    print("Predicted Blood Group:", blood)
    print("Confidence: {:.2f}%".format(confidence))
    print()
    print("All class probabilities:")
    for cls, prob in zip(classes, prediction[0]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:4s}: {prob*100:5.1f}%  {bar}")

except Exception as e:
    print("❌ Error:", e)