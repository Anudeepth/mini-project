import time
import serial.tools.list_ports
import tensorflow as tf
import numpy as np
import cv2
from pyfingerprint.pyfingerprint import PyFingerprint


# -----------------------------
# Blood group labels
# -----------------------------
classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']


# -----------------------------
# Detect scanner port
# -----------------------------
ports = [p.device for p in serial.tools.list_ports.comports()]

port = None
for p in ports:
    if "USB" in p.upper() or "ACM" in p.upper():
        port = p
        break

if port is None:
    print("❌ Fingerprint scanner not detected")
    exit()

print("✅ Scanner detected on:", port)


try:

    # -----------------------------
    # Connect to scanner
    # -----------------------------
    f = PyFingerprint(port, 57600, 0xFFFFFFFF, 0x00000000)

    if not f.verifyPassword():
        raise ValueError("Scanner password incorrect")

    print("✅ Scanner initialized")


    # -----------------------------
    # Wait for finger
    # -----------------------------
    print("Place your finger on the scanner...")

    while not f.readImage():
        time.sleep(0.2)

    print("✅ Fingerprint captured")


    # -----------------------------
    # Save image from scanner
    # -----------------------------
    img_path = "/home/anudeepth/Documents/mini-project/mini-project/fingerprint_raw.bmp"

    f.downloadImage(img_path)

    print("✅ Image saved:", img_path)


    # -----------------------------
    # Load trained AI model
    # -----------------------------
    print("Loading model...")

    model = tf.keras.models.load_model("fingerprint_model.keras")

    print("✅ Model loaded")
    print("Model input shape:", model.input_shape)


    # -----------------------------
    # Preprocessing
    # -----------------------------
    def preprocess_fingerprint(path):

        img = cv2.imread(path)

        if img is None:
            raise ValueError("Fingerprint image could not be loaded")

        # Convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        r,g,b = cv2.split(img)

        r = clahe.apply(r)
        g = clahe.apply(g)
        b = clahe.apply(b)

        img = cv2.merge((r,g,b))


        # Reduce noise
        img = cv2.GaussianBlur(img,(3,3),0)


        # Enhance fingerprint ridges
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

        gray = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)


        # Resize to model input
        gray = cv2.resize(gray,(128,128))


        # Normalize
        gray = gray.astype("float32") / 255.0


        # Expand dims
        gray = np.expand_dims(gray,axis=0)

        gray = np.repeat(gray[...,np.newaxis],3,axis=-1)


        return gray


    # -----------------------------
    # Run preprocessing
    # -----------------------------
    img = preprocess_fingerprint(img_path)


    # -----------------------------
    # AI prediction
    # -----------------------------
    prediction = model.predict(img)

    index = np.argmax(prediction)

    blood = classes[index]

    confidence = prediction[0][index] * 100


    # -----------------------------
    # Print result
    # -----------------------------
    print("\n========== RESULT ==========")

    print("Predicted Blood Group:", blood)

    print("Confidence: {:.2f}%".format(confidence))


except Exception as e:

    print("❌ Error:", e)