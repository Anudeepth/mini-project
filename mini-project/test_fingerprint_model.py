# test_fingerprint.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -----------------------------
# Load your trained model
# -----------------------------
MODEL_PATH = "fingerprint_model.keras"  # or .keras if you saved that way
model = load_model(MODEL_PATH)

# Your class names must match training
classes = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_fingerprint(img_path, input_shape=(128,128,1)):
    """
    Reads an image, preprocesses to match model training:
    - grayscale
    - resize
    - contrast enhancement
    - noise reduction
    - normalize
    - add batch dimension
    """
    img = cv2.imread(img_path)

    # Convert to grayscale if model expects 1 channel
    if input_shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to model input size
    img = cv2.resize(img, (input_shape[0], input_shape[1]))

    # Improve contrast
    img = cv2.equalizeHist(img) if input_shape[2]==1 else img

    # Noise reduction
    img = cv2.GaussianBlur(img,(3,3),0)

    # Normalize
    img = img / 255.0

    # Add channel dimension if needed
    if input_shape[2] == 1:
        img = np.expand_dims(img, axis=-1)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

# -----------------------------
# Prediction function
# -----------------------------
def predict_blood_group(img_path):
    input_shape = model.input_shape[1:]  # exclude batch dimension
    img = preprocess_fingerprint(img_path, input_shape=input_shape)

    pred = model.predict(img)
    index = np.argmax(pred)
    blood = classes[index]
    confidence = pred[0][index] * 100

    print(f"Blood Group Prediction: {blood} ({confidence:.2f}%)")

# -----------------------------
# Main test
# -----------------------------
if __name__ == "__main__":
    # Path to fingerprint image saved from your scanner
    IMAGE_PATH = "fingerprint.bmp"
    predict_blood_group(IMAGE_PATH)