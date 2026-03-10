import tensorflow as tf
import cv2
import numpy as np
import os

MODEL_PATH = "fingerprint_model.h5"
IMAGE_PATH = "test_fingerprint.png"

print("------ MODEL TEST START ------")

# 1 Load model
print("\nLoading model...")
model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)

print("Model loaded successfully")

# 2 Model summary
print("\nModel Summary:")
model.summary()

# 3 Input shape
print("\nModel Input Shape:")
print(model.input_shape)

# 4 Load image
print("\nLoading image:", IMAGE_PATH)

if not os.path.exists(IMAGE_PATH):
    print("ERROR: Image file not found")
    exit()

img = cv2.imread(IMAGE_PATH)

print("\nOriginal Image Shape:", img.shape)

# 5 Convert color
print("\nConverting BGR to RGB")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 6 Resize according to model
height = model.input_shape[1]
width = model.input_shape[2]

print("\nResizing image to:", width, "x", height)
img = cv2.resize(img, (width, height))

print("Resized Shape:", img.shape)

# 7 Normalize
print("\nNormalizing image (divide by 255)")
img = img / 255.0

# 8 Add batch dimension
img = np.expand_dims(img, axis=0)

print("Final Input Shape:", img.shape)

# 9 Predict
print("\nRunning prediction...")
prediction = model.predict(img)

print("\nPrediction Raw Output:")
print(prediction)

# 10 Predicted class
predicted_class = np.argmax(prediction)

print("\nPredicted Class Index:", predicted_class)
print("Confidence:", prediction[0][predicted_class])

print("\n------ TEST COMPLETE ------")