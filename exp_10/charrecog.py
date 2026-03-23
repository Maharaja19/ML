import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# ----------------------------------
# Model path
# ----------------------------------
model_path = "character_model.keras"

# ----------------------------------
# Load EMNIST Dataset
# ----------------------------------
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/byclass",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

# ----------------------------------
# Preprocessing
# ----------------------------------
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image,(28,28,1))
    return image, label

ds_train = ds_train.map(preprocess).shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

# ----------------------------------
# Train model if not exists
# ----------------------------------
if not os.path.exists(model_path):

    print("Training Improved CNN Model...")

    model = keras.Sequential([

        keras.Input(shape=(28,28,1)),

        layers.Conv2D(32,(3,3),padding="same",activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64,(3,3),padding="same",activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128,(3,3),padding="same",activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),

        layers.Dense(256,activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(62,activation="softmax")  # 62 characters
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        ds_train,
        epochs=15,
        validation_data=ds_test
    )

    model.save(model_path)

else:
    print("Loading saved model...")
    model = keras.models.load_model(model_path)

# ----------------------------------
# Evaluate model
# ----------------------------------
loss, acc = model.evaluate(ds_test)
print("Test Accuracy:", acc)

# ----------------------------------
# Character labels
# ----------------------------------
characters = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

# ----------------------------------
# Predict custom image
# ----------------------------------
image_path = input("Enter image path: ")

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found")
    exit()

# Resize
img = cv2.resize(img,(28,28))

# Fix EMNIST orientation
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
img = cv2.flip(img,1)

# Convert to white text on black
_, img = cv2.threshold(img,128,255,cv2.THRESH_BINARY_INV)

# Normalize
img = img.astype("float32")/255.0
img = img.reshape(1,28,28,1)

prediction = model.predict(img)

predicted_class = np.argmax(prediction)

print("Predicted Character:", characters[predicted_class])

# ----------------------------------
# Show input image
# ----------------------------------
plt.imshow(img.reshape(28,28), cmap="gray")
plt.title(f"Prediction: {characters[predicted_class]}")
plt.show()