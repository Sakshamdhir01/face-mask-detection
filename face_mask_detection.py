
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))




DATA_DIR = "data/face-mask-dataset"   

IMG_SIZE   = (64, 64)  
BATCH_SIZE = 16
EPOCHS_CNN = 20
EPOCHS_TL  = 10
SEED       = 42



CLASS_NAMES = ['with_mask', 'without_mask']


# STEP 3: DATASET LOADING & EXPLORATION


def load_dataset(data_dir, img_size):
    """Load all images and labels from class sub-folders."""
    images, labels = [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            print(f"[WARNING] Folder not found: {class_path}")
            continue
        files = os.listdir(class_path)
        print(f"  {class_name}: {len(files)} images")
        for fname in files:
            fpath = os.path.join(class_path, fname)
            try:
                img = Image.open(fpath).convert('RGB').resize(img_size)
                images.append(np.array(img))
                labels.append(label_idx)
            except Exception as e:
                print(f"  [SKIP] {fname}: {e}")
    return np.array(images), np.array(labels)

print("\n Loading dataset...")
X, y = load_dataset(DATA_DIR, IMG_SIZE)
print(f"\nDataset loaded: {X.shape[0]} images | Shape: {X.shape}")
print(f"Class distribution → with_mask: {np.sum(y==0)} | without_mask: {np.sum(y==1)}")

# ── Plot sample images ──
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Sample Images from Dataset", fontsize=14, fontweight='bold')
for i, ax in enumerate(axes.flatten()):
    idx = np.where(y == i // 5)[0][i % 5]
    ax.imshow(X[idx])
    ax.set_title(CLASS_NAMES[y[idx]], fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig("outputs/sample_images.png", dpi=150)
plt.show()
print("Saved: outputs/sample_images.png")

# ── Class distribution bar chart ──
plt.figure(figsize=(6, 4))
counts = [np.sum(y == i) for i in range(len(CLASS_NAMES))]
bars = plt.bar(CLASS_NAMES, counts, color=['#2ecc71', '#e74c3c'], edgecolor='black')
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             str(count), ha='center', fontweight='bold')
plt.title("Class Distribution", fontsize=13, fontweight='bold')
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=150)
plt.show()
print("Saved: outputs/class_distribution.png")


# STEP 4: PREPROCESSING


# Normalize pixel values to [0, 1]
X = X.astype('float32') / 255.0

# Train / Validation / Test split  →  70% / 15% / 15%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

print(f"\nSplit → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


# STEP 5: DATA AUGMENTATION


datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.05
)
datagen.fit(X_train)

# Visualize augmented images
sample_img = X_train[0:1]
aug_iter = datagen.flow(sample_img, batch_size=1)
fig, axes = plt.subplots(1, 6, figsize=(14, 3))
axes[0].imshow(sample_img[0])
axes[0].set_title("Original")
axes[0].axis('off')
for i in range(1, 6):
    aug = next(aug_iter)[0]
    axes[i].imshow(aug)
    axes[i].set_title(f"Aug #{i}")
    axes[i].axis('off')
plt.suptitle("Data Augmentation Examples", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/augmentation_examples.png", dpi=150)
plt.show()
print("Saved: outputs/augmentation_examples.png")


# STEP 6: BUILD CUSTOM CNN MODEL


def build_cnn(input_shape=(128, 128, 3)):
    """Custom CNN architecture."""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.40),

        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.50),
        layers.Dense(1, activation='sigmoid')   # Binary classification
    ], name="Custom_CNN")
    return model

cnn_model = build_cnn(input_shape=(*IMG_SIZE, 3))
cnn_model.summary()

cnn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
os.makedirs("outputs", exist_ok=True)
cnn_callbacks = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    callbacks.ModelCheckpoint("outputs/best_cnn_model.keras", save_best_only=True, verbose=1)
]


# STEP 7: TRAIN CUSTOM CNN


print("\n Training Custom CNN...")
cnn_history = cnn_model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS_CNN,
    validation_data=(X_val, y_val),
    callbacks=cnn_callbacks,
    verbose=1
)


# STEP 8: EVALUATE CUSTOM CNN


def plot_history(history, model_name, save_path):
    """Plot accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{model_name} — Training Curves", fontsize=14, fontweight='bold')

    ax1.plot(history.history['accuracy'], label='Train Acc', color='#2ecc71', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Acc', color='#27ae60',
             linestyle='--', linewidth=2)
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train Loss', color='#e74c3c', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', color='#c0392b',
             linestyle='--', linewidth=2)
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")

plot_history(cnn_history, "Custom CNN", "outputs/cnn_training_curves.png")

def evaluate_model(model, X_test, y_test, model_name):
    """Full evaluation: accuracy, confusion matrix, classification report."""
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*50}")
    print(f"  {model_name} — Test Results")
    print(f"{'='*50}")
    print(f"  Test Loss    : {loss:.4f}")
    print(f"  Test Accuracy: {acc*100:.2f}%")

    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor='gray')
    plt.title(f"{model_name} — Confusion Matrix", fontweight='bold')
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = f"outputs/confusion_matrix_{model_name.replace(' ','_')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")

    return acc, loss, y_pred

cnn_acc, cnn_loss, cnn_preds = evaluate_model(cnn_model, X_test, y_test, "Custom CNN")


# STEP 9: TRANSFER LEARNING — MobileNetV2


print("\n Building Transfer Learning model (MobileNetV2)...")

try:
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    print("MobileNetV2 loaded with ImageNet weights ✅")
except Exception as e:
    print(f"Could not download weights, using random init: {e}")
    base_model = MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
base_model.trainable = False   # Freeze pretrained layers

tl_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.40),
    layers.Dense(1, activation='sigmoid')
], name="MobileNetV2_Transfer")

tl_model.summary()

tl_model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

tl_callbacks = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    callbacks.ModelCheckpoint("outputs/best_tl_model.keras", save_best_only=True, verbose=1)
]

print("\n Training MobileNetV2 Transfer Learning model...")
tl_history = tl_model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS_TL,
    validation_data=(X_val, y_val),
    callbacks=tl_callbacks,
    verbose=1
)

plot_history(tl_history, "MobileNetV2 Transfer Learning", "outputs/tl_training_curves.png")
tl_acc, tl_loss, tl_preds = evaluate_model(tl_model, X_test, y_test, "MobileNetV2")


# STEP 10: MODEL COMPARISON TABLE


print("\n\n" + "="*55)
print("           MODEL COMPARISON SUMMARY")
print("="*55)
print(f"  {'Model':<30} {'Test Acc':>10} {'Test Loss':>10}")
print("-"*55)
print(f"  {'Custom CNN':<30} {cnn_acc*100:>9.2f}% {cnn_loss:>10.4f}")
print(f"  {'MobileNetV2 (Transfer)':<30} {tl_acc*100:>9.2f}% {tl_loss:>10.4f}")
print("="*55)

# Bar chart comparison
fig, ax = plt.subplots(figsize=(7, 5))
models_names = ['Custom CNN', 'MobileNetV2\n(Transfer Learning)']
accuracies = [cnn_acc * 100, tl_acc * 100]
colors = ['#3498db', '#9b59b6']
bars = ax.bar(models_names, accuracies, color=colors, edgecolor='black', width=0.4)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{acc:.2f}%", ha='center', fontweight='bold', fontsize=11)
ax.set_ylim(0, 105)
ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title("Model Comparison — Test Accuracy", fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
plt.show()
print("Saved: outputs/model_comparison.png")


# STEP 11: PREDICTION ON SAMPLE TEST IMAGES


def predict_samples(model, X_test, y_test, model_name, n=10):
    """Visualize predictions on sample test images."""
    indices = np.random.choice(len(X_test), n, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle(f"{model_name} — Sample Predictions", fontsize=13, fontweight='bold')
    for i, ax in enumerate(axes.flatten()):
        idx = indices[i]
        img = X_test[idx]
        actual = CLASS_NAMES[y_test[idx]]
        prob = model.predict(img[np.newaxis, ...], verbose=0)[0][0]
        predicted = CLASS_NAMES[int(prob > 0.5)]
        color = 'green' if actual == predicted else 'red'
        ax.imshow(img)
        ax.set_title(f"A: {actual}\nP: {predicted}", color=color, fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    fname = f"outputs/predictions_{model_name.replace(' ','_')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")

predict_samples(cnn_model,  X_test, y_test, "Custom CNN")
predict_samples(tl_model,   X_test, y_test, "MobileNetV2")


# STEP 12: SAVE MODELS


cnn_model.save("outputs/cnn_model_final.keras")
tl_model.save("outputs/tl_model_final.keras")
print("\nModels saved to outputs/")
print("\n All done! Check the 'outputs/' folder for all figures and saved models.")
