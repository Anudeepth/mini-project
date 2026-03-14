import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2

train_dir = "dataset" # We will use the full dataset folder
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Load dataset
print("Loading dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names
print("Classes identified:", class_names)

# ResNet50V2 expects inputs from [-1, 1], so we normalize accordingly
# The native preprocessing layer for ResNet50V2 handles this:
preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

# Data Augmentation to prevent overfitting on such a complex model
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Create the base model from the pre-trained model ResNet50V2
# We set include_top=False to discard the 1000-class ImageNet head
base_model = ResNet50V2(input_shape=(128, 128, 3),
                        include_top=False,
                        weights='imagenet')

# FINE-TUNING:
# Unfreeze the base model so we can alter its core weights
base_model.trainable = True

# We want to freeze the very bottom layers (which detect simple lines) 
# and only train the top ~30 layers (which detect complex fingerprint loops/whorls).
fine_tune_at = len(base_model.layers) - 30

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Build the complete model
inputs = keras.Input(shape=(128, 128, 3))
x = data_augmentation(inputs)

# Standard preprocessing for ResNet
x = preprocess_input(x)

# Pass through the frozen ResNet50V2
x = base_model(x, training=False) 

# Convert the features to a single vector per image
x = layers.GlobalAveragePooling2D()(x)

# Add a dropout layer to fight overfitting
x = layers.Dropout(0.2)(x)

# Create our custom output layer with 8 neurons (one for each blood group)
outputs = layers.Dense(8, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# Compile the model
# Using a very low learning rate for fine-tuning so we don't destroy the pre-trained weights
print("Compiling model for Fine-Tuning...")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks: Automate the training process so it perfectly stops when accuracy peaks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# Train the top layer
print("Training custom classification head & Top 30 ResNet Layers...")
history = model.fit(train_ds,
                    epochs=20, # Let EarlyStopping decide when to quit
                    validation_data=val_ds,
                    callbacks=[early_stopping, reduce_lr])

# Save the mighty new brain
model.save("resnet_fingerprint_model.keras")
print("\nSuccess! Saved fine-tuned maximum-accuracy model to resnet_fingerprint_model.keras")
