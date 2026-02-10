"""
Dog Skin Disease Detection — Optimized Model Training
======================================================
Uses MobileNetV2 transfer learning (fast on CPU) with:
  - Data augmentation
  - Class weight balancing
  - Learning rate scheduling
  - Early stopping + model checkpointing
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter

# ══════════════════════════════════
# Configuration
# ══════════════════════════════════
DATASET_PATH = r"C:\Users\pawan raj\Downloads\dog disease dataset"
OUTPUT_DIR = r"C:\Users\pawan raj\Desktop\A dog disease"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.h5")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
FINE_TUNE_EPOCHS = 20
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5

print("=" * 60)
print("  Dog Skin Disease Detection — Model Training")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
if not tf.config.list_physical_devices('GPU'):
    print("  Training on CPU — using MobileNetV2 (lightweight)")
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print()

# ══════════════════════════════════
# Data Loading with Augmentation
# ══════════════════════════════════

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='reflect'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

print("Loading datasets...")
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'valid'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

CLASS_NAMES = list(train_generator.class_indices.keys())
NUM_CLASSES = len(CLASS_NAMES)
print(f"\nFound {NUM_CLASSES} classes: {CLASS_NAMES}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# ══════════════════════════════════
# Compute Class Weights
# ══════════════════════════════════
class_counts = Counter(train_generator.classes)
total_samples = sum(class_counts.values())
class_weights = {}
print("\nClass distribution & weights:")
for cls_idx, count in sorted(class_counts.items()):
    weight = total_samples / (NUM_CLASSES * count)
    class_weights[cls_idx] = weight
    print(f"  {CLASS_NAMES[cls_idx]:25s}: {count:5d} imgs, weight={weight:.3f}")

# Save class names
class_info_path = os.path.join(OUTPUT_DIR, "class_names.json")
with open(class_info_path, 'w') as f:
    json.dump({
        'class_names': CLASS_NAMES,
        'class_indices': train_generator.class_indices,
        'img_size': IMG_SIZE
    }, f, indent=2)
print(f"\nSaved class names to {class_info_path}")

# ══════════════════════════════════
# Build Model — MobileNetV2
# ══════════════════════════════════
print("\nBuilding MobileNetV2 model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs, outputs)
print(f"Model parameters: {model.count_params():,}")

# ══════════════════════════════════
# Phase 1: Train Classification Head
# ══════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 1: Training classification head (base frozen)")
print("=" * 60)

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

phase1_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_accuracy', patience=7,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-6, verbose=1
    ),
    callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, monitor='val_accuracy',
        save_best_only=True, verbose=1
    )
]

history1 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=phase1_callbacks,
    verbose=1
)

p1_acc = max(history1.history['val_accuracy'])
print(f"\nPhase 1 Best Val Accuracy: {p1_acc:.4f}")

# ══════════════════════════════════
# Phase 2: Fine-tune top layers
# ══════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning (unfreezing top 30% of base)")
print("=" * 60)

base_model.trainable = True
freeze_until = int(len(base_model.layers) * 0.7)
for layer in base_model.layers[:freeze_until]:
    layer.trainable = False
print(f"Frozen: {freeze_until}/{len(base_model.layers)} base layers")

model.compile(
    optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

phase2_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_accuracy', patience=8,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3,
        patience=3, min_lr=1e-7, verbose=1
    ),
    callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, monitor='val_accuracy',
        save_best_only=True, verbose=1
    )
]

history2 = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=phase2_callbacks,
    verbose=1
)

p2_acc = max(history2.history['val_accuracy'])
print(f"\nPhase 2 Best Val Accuracy: {p2_acc:.4f}")

# ══════════════════════════════════
# Evaluate on Test Set
# ══════════════════════════════════
print("\n" + "=" * 60)
print("Final Evaluation on Test Set")
print("=" * 60)

model = keras.models.load_model(MODEL_SAVE_PATH)
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Per-class results
from sklearn.metrics import classification_report
test_generator.reset()
preds = model.predict(test_generator, verbose=1)
pred_classes = np.argmax(preds, axis=1)
true_classes = test_generator.classes

print("\nClassification Report:")
print(classification_report(true_classes, pred_classes, target_names=CLASS_NAMES))

print(f"\n{'=' * 60}")
print(f"TRAINING COMPLETE!")
print(f"  Model saved to: {MODEL_SAVE_PATH}")
print(f"  Phase 1 Val Acc: {p1_acc:.4f}")
print(f"  Phase 2 Val Acc: {p2_acc:.4f}")
print(f"  Test Accuracy:   {test_acc:.4f}")
print(f"{'=' * 60}")
