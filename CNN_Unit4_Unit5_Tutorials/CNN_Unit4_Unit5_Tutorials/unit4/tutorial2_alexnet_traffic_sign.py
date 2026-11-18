"""
Unit 4 - Tutorial 2: Traffic Sign Classifier Using AlexNet
----------------------------------------------------------
Objective:
Implement a beginner-friendly AlexNet-style Convolutional Neural Network (CNN)
to classify traffic sign images into three categories:
- No_Entry
- Speed_Limit
- Stop

Key Components:
- Convolution layers with ReLU activation
- MaxPooling layers
- Flatten layer
- Fully connected Dense layers with Dropout regularization
- Softmax output layer for 3-class classification
"""

import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

# Set random seed for reproducibility
tf.random.set_seed(42)

# 1. Define dataset path using pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "traffic_signs"

# Verify dataset folder exists and has PNG files
if not DATASET_DIR.exists() or not any(DATASET_DIR.rglob("*.png")):
    print(f"Error: Traffic signs dataset not found at '{DATASET_DIR}'.")
    print("Please run 'python generate_datasets.py' first.")
    sys.exit(1)

# Hyperparameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 5

print("=" * 70)
print("UNIT 4 - TUTORIAL 2: Traffic Sign Classifier Using AlexNet-Style CNN")
print("=" * 70)
print(f"Loading traffic sign images from: {DATASET_DIR}")

# 2. Load training and validation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Traffic Sign Classes ({num_classes}): {class_names}")

# 3. Build a simplified AlexNet-style CNN architecture
model = models.Sequential([
    # Normalize pixel values to [0, 1]
    layers.Rescaling(1.0 / 255, input_shape=(128, 128, 3)),

    # 1st Convolutional Block (large filter size inspired by AlexNet)
    layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # 2nd Convolutional Block
    layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # 3rd, 4th, and 5th Stacked Convolutional Layers (classic AlexNet structure)
    layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Classifier Head with Flatten, Dense, and Dropout layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

# 4. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 5. Train the model
print("\nStarting training for AlexNet Traffic Sign Classifier...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# 6. Evaluate and print validation accuracy
val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
print("\n" + "-" * 50)
print(f"Traffic Sign Validation Loss     : {val_loss:.4f}")
print(f"Traffic Sign Validation Accuracy : {val_accuracy * 100:.2f}%")
print("-" * 50)
