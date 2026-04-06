import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
TRAIN_DIR = "split_dataset/train"        # ✅ Fixed: use the large 4,196-image dataset
VAL_DIR   = "split_dataset/validation"   # ✅ Fixed: use pre-split validation folder
IMG_SIZE  = (128, 128)
BATCH_SIZE = 32

# ─────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────
print("Loading dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False,
    seed=42
)

class_names = train_ds.class_names
print("Classes identified:", class_names)

# ✅ Save class names to JSON so inference always uses the exact same order
with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print("Class names saved to class_names.json")

# ─────────────────────────────────────────
# Preprocessing — ResNet50V2 expects [-1, 1]
# Applied identically at train + inference time
# ─────────────────────────────────────────
preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

# ─────────────────────────────────────────
# Data augmentation (train only)
# ─────────────────────────────────────────
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1),
], name="augmentation")

# ─────────────────────────────────────────
# Performance: cache & prefetch
# ─────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# ─────────────────────────────────────────
# Build model — ResNet50V2 backbone
# ─────────────────────────────────────────
base_model = ResNet50V2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze all backbone layers first (transfer learning phase)
base_model.trainable = False

inputs = keras.Input(shape=(128, 128, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)                        # ✅ Consistent preprocessing
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = keras.Model(inputs, outputs)

# ─────────────────────────────────────────
# Phase 1: Train classification head only
# ─────────────────────────────────────────
print("\n=== Phase 1: Training classification head ===")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

early_stop_phase1 = EarlyStopping(
    monitor='val_accuracy', patience=5,
    restore_best_weights=True, verbose=1
)

history1 = model.fit(
    train_ds,
    epochs=15,
    validation_data=val_ds,
    callbacks=[early_stop_phase1]
)

# ─────────────────────────────────────────
# Phase 2: Fine-tune top ResNet layers
# ─────────────────────────────────────────
print("\n=== Phase 2: Fine-tuning top 50 ResNet layers ===")
base_model.trainable = True

# Freeze everything except the top 50 layers
fine_tune_at = len(base_model.layers) - 50
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),   # ✅ Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop_phase2 = EarlyStopping(
    monitor='val_accuracy', patience=5,
    restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5,
    patience=3, min_lr=1e-6, verbose=1
)

history2 = model.fit(
    train_ds,
    epochs=30,
    validation_data=val_ds,
    callbacks=[early_stop_phase2, reduce_lr]
)

# ─────────────────────────────────────────
# Save model
# ─────────────────────────────────────────
model.save("resnet_fingerprint_model.keras")
print("\n✅ Saved fine-tuned model to resnet_fingerprint_model.keras")

# ─────────────────────────────────────────
# Final evaluation
# ─────────────────────────────────────────
loss, acc = model.evaluate(val_ds, verbose=0)
print(f"✅ Final Validation Accuracy: {acc*100:.2f}%")
