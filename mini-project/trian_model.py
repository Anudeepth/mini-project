import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

train_dir = "split_dataset/train"
val_dir = "split_dataset/validation"

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(128,128),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(128,128),
    batch_size=32
)

class_names = train_ds.class_names
print("Classes:", class_names)

normalization = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x,y:(normalization(x),y))
val_ds = val_ds.map(lambda x,y:(normalization(x),y))

model = keras.Sequential([

    layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(256,(3,3),activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(256,activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(8,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

model.save("fingerprint_model.keras")

print("Model saved as fingerprint_model.keras")