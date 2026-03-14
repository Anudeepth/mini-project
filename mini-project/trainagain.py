import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Load existing model
# -----------------------------
model = tf.keras.models.load_model("fingerprint_model.keras")

# -----------------------------
# Dataset preprocessing
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    "dataset",
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val = datagen.flow_from_directory(
    "dataset",
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# -----------------------------
# Compile again
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Continue training
# -----------------------------
history = model.fit(
    train,
    validation_data=val,
    epochs=10
)

# -----------------------------
# Save improved model
# -----------------------------
model.save("fingerprint_model_improved.keras")