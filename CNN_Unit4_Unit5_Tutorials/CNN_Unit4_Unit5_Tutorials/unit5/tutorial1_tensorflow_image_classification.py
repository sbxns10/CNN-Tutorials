"""
Unit 5 - Tutorial 1: Implement Image Classification with TensorFlow
-------------------------------------------------------------------
Objective:
Build a basic Convolutional Neural Network (CNN) using TensorFlow/Keras on
the general_images dataset (Apple, Banana, Orange), evaluate it on a test split,
and display:
- Test loss
- Test accuracy
- Predicted class vs Actual class on sample test images
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# 1. Define dataset path using pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "general_images"

# Check that dataset directory exists and has PNG files
if not DATASET_DIR.exists() or not any(DATASET_DIR.rglob("*.png")):
    print(f"Error: General images dataset not found at '{DATASET_DIR}'.")
    print("Please run 'python generate_datasets.py' first.")
    sys.exit(1)

# Hyperparameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 5

print("=" * 70)
print("UNIT 5 - TUTORIAL 1: Image Classification with TensorFlow")
print("=" * 70)
print(f"Loading images from: {DATASET_DIR}")

# 2. Load training and test datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# Store class names before applying normalization
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Class Names ({num_classes}): {class_names}")

# 3. Image Normalization: Scale pixel values from [0, 255] to [0.0, 1.0]
normalization_layer = layers.Rescaling(1.0 / 255)
train_ds_norm = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds_norm = test_ds.map(lambda x, y: (normalization_layer(x), y))

# 4. Build a minimal CNN using Conv2D, MaxPooling2D, Flatten, Dense, and Softmax
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

# 5. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 6. Train the model
print("\nStarting training...")
model.fit(
    train_ds_norm,
    validation_data=test_ds_norm,
    epochs=EPOCHS,
)

# 7. Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_ds_norm, verbose=0)
print("\n" + "-" * 50)
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print("-" * 50)

# 8. Predict on sample test images and display Predicted Class vs Actual Class
print("\nSample Predictions on Test Images:")
for images, labels in test_ds_norm.take(1):
    predictions = model.predict(images, verbose=0)
    num_samples = min(5, len(labels))
    for i in range(num_samples):
        predicted_idx = int(np.argmax(predictions[i]))
        actual_idx = int(labels[i].numpy())
        predicted_class = class_names[predicted_idx]
        actual_class = class_names[actual_idx]
        print(f"  Sample {i + 1}: Predicted Class = {predicted_class:<8} | Actual Class = {actual_class}")
