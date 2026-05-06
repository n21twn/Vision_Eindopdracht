import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Paden instellen
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(BASE_DIR, "object_crops_split_2")
 
IMG_SIZE   = 224
BATCH_SIZE = 16  # Lager voor CPU stabiliteit

# Data Generators
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    brightness_range=[0.6, 1.4],
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False
)

val_gen = ImageDataGenerator(rescale=1./255)
 
# Data inladen
train_data = train_gen.flow_from_directory(
    os.path.join(SPLIT_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    os.path.join(SPLIT_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Model opbouw (MobileNetV2 Transfer Learning)
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False 

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3), 
    Dense(128, activation="relu"),
    Dense(train_data.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# EarlyStopping om tijd te besparen op de CPU
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training starten
print("\nStart training op CPU...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50, 
    callbacks=[early_stop],
    verbose=1 # Toont progressie per epoch (balkje)
)

# --- VISUALISATIE VAN DE PROGRESSIE ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

# Grafiek 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Progressie: Accuracy')
plt.legend(loc='lower right')

# Grafiek 2: Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Progressie: Loss')
plt.legend(loc='upper right')

plt.show() # Dit opent een venster met de grafieken

# Opslaan
model.save(os.path.join(BASE_DIR, "poker_model_cpu_2.keras"))
print("\nModel en grafieken zijn klaar!")