"""
train_and_save.py
Run this ONCE to train the model and save it.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test  = X_test  / 255.0

print("Building model...")
model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\nTraining... (this takes ~1 minute)")
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=1)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

model.save("mnist_model.keras")
print("✅ Model saved as mnist_model.keras")
