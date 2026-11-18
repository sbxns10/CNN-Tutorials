# Deep Learning Lab Tutorials — Unit 4 & Unit 5

## A. Project Introduction

This repository contains a complete, self-contained collection of all **six Deep Learning lab tutorials** covering **Unit 4 (CNN Architectures & Transfer Learning)** and **Unit 5 (TensorFlow Image Classification, Input Pipelines & Regularization)**.

The project is specifically designed for college lab submissions and evaluations:
- **Beginner-friendly Python code** with clear step-by-step comments.
- **Pre-generated PNG image datasets** included directly inside the `datasets/` folder (288 valid RGB PNG images across 8 class folders, 36 images per class), so no external dataset downloads are required.
- **Cross-directory execution support** using Python's `pathlib`, allowing every script to run reliably from the project root or any subdirectory.

### Project Directory Structure

```text
DL_Unit4_Unit5_Tutorials/
│
├── README.md
├── requirements.txt
├── generate_datasets.py
│
├── unit4/
│   ├── tutorial1_imagenet_resnet50.py
│   ├── tutorial2_alexnet_traffic_sign.py
│   └── tutorial3_vggnet.py
│
├── unit5/
│   ├── tutorial1_tensorflow_image_classification.py
│   ├── tutorial2_tensorflow_input_pipeline.py
│   └── tutorial3_cnn_l2_regularization.py
│
└── datasets/
    ├── general_images/
    │   ├── Apple/        (36 RGB PNG images)
    │   ├── Banana/       (36 RGB PNG images)
    │   └── Orange/       (36 RGB PNG images)
    │
    ├── traffic_signs/
    │   ├── No_Entry/     (36 RGB PNG images)
    │   ├── Speed_Limit/  (36 RGB PNG images)
    │   └── Stop/         (36 RGB PNG images)
    │
    └── cats_dogs/
        ├── Cats/         (36 RGB PNG images)
        └── Dogs/         (36 RGB PNG images)
```

---

## B. Requirements

- **Operating System:** Windows 10 / Windows 11 (64-bit) (also compatible with macOS and Linux)
- **Python Version:** Python 3.11 (64-bit)
- **Core Framework:** TensorFlow 2.15.1 (includes Keras)
- **Required Python Packages (listed in `requirements.txt`):**
  - `tensorflow==2.15.1`
  - `numpy==1.26.4`
  - `matplotlib==3.8.4`
  - `Pillow==10.3.0`
- **Internet Connection Note:** All three image datasets are already pre-generated and included inside the `datasets/` folder. However, **internet access is required on the first run of Unit 4 Tutorial 1 (`unit4/tutorial1_imagenet_resnet50.py`)** in normal mode so Keras can download the pretrained ResNet50 ImageNet weights (`weights="imagenet"`). If no internet connection is available, Tutorial 1 automatically falls back to `weights=None` and prints a clear warning message instead of crashing.

---

## C. Virtual Environment Setup

Using a single shared virtual environment (`.venv`) for all six tutorials keeps your Python installation clean and prevents package version conflicts.

### Option 1: Windows PowerShell

Open PowerShell inside the `DL_Unit4_Unit5_Tutorials` project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Windows Command Prompt (CMD)

Open Command Prompt (`cmd.exe`) inside the `DL_Unit4_Unit5_Tutorials` project folder and run:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## D. How to Execute

Once your virtual environment is activated, run any of the six tutorials directly from the `DL_Unit4_Unit5_Tutorials` root folder:

```powershell
python unit4/tutorial1_imagenet_resnet50.py
python unit4/tutorial2_alexnet_traffic_sign.py
python unit4/tutorial3_vggnet.py
python unit5/tutorial1_tensorflow_image_classification.py
python unit5/tutorial2_tensorflow_input_pipeline.py
python unit5/tutorial3_cnn_l2_regularization.py
```

*(Optional)* The sample PNG images are already included in `datasets/`. If you ever want to regenerate them from scratch, run:

```powershell
python generate_datasets.py
```

---

## E. Tutorial Explanations

### 1. Unit 4 — Tutorial 1: Implement ImageNet Model for a Dataset (`unit4/tutorial1_imagenet_resnet50.py`)
- **Objective:** Perform transfer learning using a ResNet50 model pretrained on ImageNet to classify fruit images.
- **Model / Technique Used:** `ResNet50(weights="imagenet", include_top=False)` with frozen convolutional base layers (`base_model.trainable = False`), ` preprocess_input`, `GlobalAveragePooling2D`, a `Dense(64, activation="relu")` layer, and a 3-class ` softmax` output layer. Includes automatic fallback to `weights=None` if offline.
- **Dataset Used:** `datasets/general_images/` (`Apple`, `Banana`, `Orange`), automatically resized to `224 × 224`.
- **Expected Output:** Displays detected class names, model summary, epoch-by-epoch training/validation progress (4 epochs), and final validation loss and validation accuracy.

### 2. Unit 4 — Tutorial 2: Traffic Sign Classifier Using AlexNet (`unit4/tutorial2_alexnet_traffic_sign.py`)
- **Objective:** Classify traffic sign illustrations into three categories using an AlexNet-inspired CNN architecture.
- **Model / Technique Used:** Simplified AlexNet architecture featuring an initial `5 × 5` `Conv2D` layer, `MaxPooling2D` layers, three stacked `3 × 3` `Conv2D` layers with `relu` activation, `Flatten`, two `Dense` layers with `Dropout(0.5)`, and a `softmax` output layer.
- **Dataset Used:** `datasets/traffic_signs/` (`No_Entry`, `Speed_Limit`, `Stop`), image size `128 × 128`.
- **Expected Output:** Displays traffic sign classes, model architecture summary, 5 epochs of training logs, and final validation loss and accuracy.

### 3. Unit 4 — Tutorial 3: Implement VGGNet Model for a Dataset (`unit4/tutorial3_vggnet.py`)
- **Objective:** Implement a VGGNet-style CNN using uniform `3 × 3` convolutional blocks for multi-class image classification.
- **Model / Technique Used:** Three VGG blocks, each containing two consecutive `3 × 3` `Conv2D` layers (`relu` activation, `padding="same"`) followed by `MaxPooling2D(2, 2)`, connected to `Flatten`, `Dense(128)`, `Dense(64)`, and a `softmax` classification layer.
- **Dataset Used:** `datasets/general_images/` (`Apple`, `Banana`, `Orange`), image size `128 × 128`.
- **Expected Output:** Displays model summary, training logs across 5 epochs, and final validation loss and validation accuracy.

### 4. Unit 5 — Tutorial 1: Implement Image Classification with TensorFlow (`unit5/tutorial1_tensorflow_image_classification.py`)
- **Objective:** Build, train, and evaluate a foundational TensorFlow/Keras CNN and compare predicted vs. actual labels on test images.
- **Model / Technique Used:** Image loading with `image_dataset_from_directory`, pixel normalization to `[0, 1]` using `Rescaling(1./255)`, three `Conv2D` + `MaxPooling2D` blocks, `Flatten`, `Dense`, and `softmax` classification.
- **Dataset Used:** `datasets/general_images/` (`Apple`, `Banana`, `Orange`), image size `128 × 128`.
- **Expected Output:** Prints model summary, 5-epoch training history, test loss, test accuracy, and a sample table showing `Predicted Class` vs. `Actual Class` for 5 test images.

### 5. Unit 5 — Tutorial 2: Build TensorFlow Input Pipelines for Images (`unit5/tutorial2_tensorflow_input_pipeline.py`)
- **Objective:** Demonstrate efficient `tf.data` input pipelines for loading, caching, shuffling, batching, and prefetching image datasets before training a CNN.
- **Model / Technique Used:** `image_dataset_from_directory()` with an 80/20 train/validation split, storing `class_names` prior to transformations, inspecting batch shapes (`(16, 128, 128, 3)`), and chaining `.cache()`, `.shuffle(100)`, and `.prefetch(buffer_size=tf.data.AUTOTUNE)`.
- **Dataset Used:** `datasets/cats_dogs/` (`Cats`, `Dogs`), image size `128 × 128`.
- **Expected Output:** Prints stored class names (`['Cats', 'Dogs']`), image batch shape, labels batch shape, pipeline confirmation message, CNN training progress across 5 epochs, and final evaluation loss and accuracy.

### 6. Unit 5 — Tutorial 3: Implement a CNN for Image Processing Using L2 Regularization (`unit5/tutorial3_cnn_l2_regularization.py`)
- **Objective:** Reduce overfitting in a CNN by applying L2 weight decay regularization (`l2(0.001)`) to convolutional and dense kernels.
- **Model / Technique Used:** `Conv2D` and `Dense` layers configured with `kernel_regularizer=tf.keras.regularizers.l2(0.001)`, trained with the `Adam` optimizer, `sparse_categorical_crossentropy` loss, and `accuracy` metric.
- **Dataset Used:** `datasets/general_images/` (`Apple`, `Banana`, `Orange`), image size `128 × 128`.
- **Expected Output:** Displays model summary (including regularization loss component during training), 5 epochs of training logs, and final L2-regularized test loss and test accuracy.

---

## F. Troubleshooting

1. **PowerShell Execution Policy Error When Activating `.venv`**
   - *Symptom:* `Activate.ps1 cannot be loaded because running scripts is disabled on this system.`
   - *Fix:* Run the following command in PowerShell to allow local scripts for the current user, then activate again:
     ```powershell
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     .\.venv\Scripts\Activate.ps1
     ```
   - Alternatively, use Command Prompt (`cmd.exe`) and run `.venv\Scripts\activate.bat`.

2. **Python Version Incompatibility (`No matching distribution found for tensorflow==2.15.1`)**
   - *Symptom:* `pip install -r requirements.txt` fails to find TensorFlow 2.15.1.
   - *Fix:* TensorFlow 2.15.1 requires 64-bit Python 3.9–3.11 (Python 3.12+ uses newer TensorFlow releases). Verify your Python version with `py -3.11 --version` and ensure your virtual environment was created using `py -3.11 -m venv .venv`.

3. **TensorFlow DLL Load Failed on Windows**
   - *Symptom:* `ImportError: DLL load failed while importing _pywrap_tensorflow_internal`.
   - *Fix:* Install the latest **Microsoft Visual C++ Redistributable (x64)** from Microsoft's official website and restart your terminal.

4. **Missing Image Directories or Empty Dataset Error**
   - *Symptom:* A tutorial prints `Error: Dataset not found or empty...`
   - *Fix:* Run `python generate_datasets.py` from the project root to recreate all 288 PNG images inside `datasets/`.

5. **ResNet50 ImageNet Weights Failing to Download (No Internet / Firewall)**
   - *Symptom:* Unit 4 Tutorial 1 prints `[WARNING] Could not download ImageNet weights` when running offline or behind a restricted college firewall.
   - *Fix:* No action is needed for the script to complete—it automatically switches to `[FALLBACK] Initializing ResNet50 architecture with weights=None` so your lab demonstration still runs end-to-end. To use pretrained weights, connect to the internet on the first run so Keras caches `resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5` in `~/.keras/models/`.
