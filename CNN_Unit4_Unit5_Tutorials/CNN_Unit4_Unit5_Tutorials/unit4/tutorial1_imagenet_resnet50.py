"""
Unit 4 - Tutorial 1: Implement ImageNet Model for a Dataset (ResNet50 Transfer Learning)
----------------------------------------------------------------------------------------
Objective:
Use a ResNet50 model pretrained on ImageNet to classify images from the
general_images dataset (Apple, Banana, Orange) using Transfer Learning.

Key Steps:
1. Load images from datasets/general_images/ resized to 224 x 224.
2. Apply ResNet50 ImageNet preprocessing.
3. Load ResNet50 with include_top=False and weights="imagenet"
   (includes automatic fallback to weights=None if offline).
4. Freeze pretrained base layers.
5. Add GlobalAveragePooling2D, Dense layer, and Softmax classification head.
6. Compile, train, and evaluate validation accuracy.
"""

import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Set random seed for reproducible results
tf.random.set_seed(42)

# 1. Define dataset path using pathlib (works from any terminal directory)
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "general_images"

# Check if dataset directory exists and contains images
if not DATASET_DIR.exists() or not any(DATASET_DIR.rglob("*.png")):
    print(f"Error: Dataset not found or empty at '{DATASET_DIR}'.")
    print("Please run 'python generate_datasets.py' first to create sample images.")
    sys.exit(1)

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 4

print("=" * 70)
print("UNIT 4 - TUTORIAL 1: ResNet50 Transfer Learning on General Images")
print("=" * 70)
print(f"Loading dataset from: {DATASET_DIR}")

# 2. Load training and validation datasets (resized to 224 x 224 for ResNet50)
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
print(f"Detected Classes ({num_classes}): {class_names}")

# 3. Apply ResNet50 ImageNet input preprocessing
train_ds = train_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y))

# 4. Load ResNet50 base model with graceful fallback if internet is unavailable
try:
    print("\nAttempting to load ResNet50 with pretrained ImageNet weights...")
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    print("Successfully loaded pretrained ImageNet weights!")
except Exception as err:
    print(f"\n[WARNING] Could not download ImageNet weights ({err}).")
    print("[FALLBACK] Initializing ResNet50 architecture with weights=None (untrained weights).")
    base_model = ResNet50(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3),
    )

# 5. Freeze the pretrained base model layers
base_model.trainable = False

# 6. Build the Transfer Learning model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

# 7. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 8. Train the model
print("\nStarting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# 9. Evaluate the model and display validation accuracy
val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
print("\n" + "-" * 50)
print(f"Final Validation Loss     : {val_loss:.4f}")
print(f"Final Validation Accuracy : {val_accuracy * 100:.2f}%")
print("-" * 50)
