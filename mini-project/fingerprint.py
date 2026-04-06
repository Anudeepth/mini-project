import json
import os
import cv2
import numpy as np
import tensorflow as tf
from pyfingerprint.pyfingerprint import PyFingerprint

# Load class names
classes_path = os.path.join(os.path.dirname(__file__), "class_names.json")
try:
    with open(classes_path) as f:
        blood_groups = json.load(f)
except:
    blood_groups = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

# Load model
model_path = os.path.join(os.path.dirname(__file__), "resnet_fingerprint_model.keras")
model = None
try:
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
except:
    pass

def scan_fingerprint(port=None):
    """
    Scans a fingerprint and predicts blood group using the ResNet model.
    """
    try:
        import serial.tools.list_ports
        if port is None:
            ports = [p.device for p in serial.tools.list_ports.comports()]
            port = next((p for p in ports if 'USB' in p.upper() or 'ACM' in p.upper() or 'COM' in p.upper()), None)
        
        if not port:
            return None
            
        f = PyFingerprint(port, 57600)
        if not f.verifyPassword():
            return None

        # Wait for finger
        while not f.readImage():
            pass

        # Save temporarily to predict
        tmp_path = "tmp_fingerprint.bmp"
        f.downloadImage(tmp_path)

        if model is None:
            return "Model not loaded"

        # Preprocess
        img = cv2.imread(tmp_path)
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

        # Predict
        prediction = model.predict(img)
        index = np.argmax(prediction)
        
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        return blood_groups[index]

    except Exception as e:
        print(f"Scan error: {e}")
        return None

if __name__ == "__main__":
    result = scan_fingerprint()
    if result:
        print(f"Detected Blood Group: {result}")
    else:
        print("Scan failed or scanner not detected.")