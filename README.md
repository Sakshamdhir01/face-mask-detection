# 😷 Face Mask Detection — CNN & Transfer Learning

A deep learning project that detects whether a person is wearing a face mask or not, using a Custom CNN and MobileNetV2 Transfer Learning model built with TensorFlow/Keras.

---

##  Project Overview

This project was built as part of a Machine Learning course assignment (Phase II). It trains two image classification models on a dataset of 7,553 face images to classify them into two categories:

-  **with_mask** — person wearing a face mask
-  **without_mask** — person not wearing a face mask

---

## 📊 Dataset

- **Source:** [Face Mask Dataset — Kaggle (omkargurav)](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- **Total Images:** 7,553
- **Classes:** 2 (with_mask: 3,725 | without_mask: 3,828)
- **Split:** 70% Train / 15% Validation / 15% Test

---

##  Models

### 1. Custom CNN
Built from scratch with 3 convolutional blocks:
- Conv2D → BatchNormalization → MaxPooling → Dropout (×3)
- Fully connected Dense(256) layer
- Sigmoid output for binary classification

### 2. MobileNetV2 (Transfer Learning)
- Pretrained MobileNetV2 base (ImageNet weights, frozen)
- GlobalAveragePooling2D → Dense(128) → BatchNormalization → Dropout
- Sigmoid output

---

##  Results

| Model | Test Accuracy | Test Loss |
|-------|:-------------:|:---------:|
| Custom CNN | **96.82%** | 0.0935 |
| MobileNetV2 (Transfer Learning) | **94.97%** | 0.1328 |

Both models achieved **94%+ accuracy** on the test set.

---

##  Project Structure

```
face-mask-project/
├── face_mask_detection.py    # Main Python code
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore
└── outputs/
    ├── sample_images.png
    ├── class_distribution.png
    ├── augmentation_examples.png
    ├── cnn_training_curves.png
    ├── tl_training_curves.png
    ├── confusion_matrix_Custom_CNN.png
    ├── confusion_matrix_MobileNetV2.png
    ├── model_comparison.png
    ├── predictions_Custom_CNN.png
    ├── predictions_MobileNetV2.png
    ├── best_cnn_model.keras
    └── best_tl_model.keras
```

> **Note:** The `data/` folder (dataset) and `venv/` are excluded from this repo via `.gitignore`.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/face-mask-detection.git
cd face-mask-detection
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) and place it as:
```
data/
└── face-mask-dataset/
    ├── with_mask/
    └── without_mask/
```

---

## ▶ How to Run

```bash
python face_mask_detection.py
```

The script will automatically:
1. Load and visualize the dataset
2. Preprocess and split images
3. Apply data augmentation
4. Train the Custom CNN (up to 20 epochs with early stopping)
5. Train MobileNetV2 Transfer Learning model
6. Evaluate both models and generate reports
7. Save all output figures and models to `outputs/`

---

##  Output Files

| File | Description |
|------|-------------|
| `sample_images.png` | Grid of sample dataset images |
| `class_distribution.png` | Bar chart of class counts |
| `augmentation_examples.png` | Sample augmented images |
| `cnn_training_curves.png` | CNN accuracy & loss over epochs |
| `tl_training_curves.png` | MobileNetV2 accuracy & loss over epochs |
| `confusion_matrix_Custom_CNN.png` | CNN confusion matrix |
| `confusion_matrix_MobileNetV2.png` | MobileNetV2 confusion matrix |
| `model_comparison.png` | Side-by-side accuracy comparison |
| `predictions_*.png` | Sample predictions with labels |

---

##  Technologies Used

- Python 3.11
- TensorFlow 2.21
- Keras
- NumPy
- Matplotlib & Seaborn
- scikit-learn
- OpenCV
- Pillow

---

## Done by

**Saksham**

## Supervision

**Prof. Raja Hashim Ali**