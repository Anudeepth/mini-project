import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("fingerprint_model.keras")

classes = ['A+','A-','AB+','AB-','B+','B-','O+','O-']

img_path = "test.BMP"

img = image.load_img(img_path, target_size=(128,128))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0) / 255.0

prediction = model.predict(img)

print("Predicted Blood Group:", classes[np.argmax(prediction)])