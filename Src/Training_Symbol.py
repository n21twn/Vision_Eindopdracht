import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# --- Paden instellen ---
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(BASE_DIR, "object_crops_split")  # map met 2 klassen: rood / zwart

IMG_SIZE   = 224
BATCH_SIZE = 32

# --- Data augmentatie ---
# Augmentatie zorgt voor meer variatie in de traindata
# zodat het model robuuster wordt voor echte foto's
train_gen = ImageDataGenerator(
    rescale=1./255,            # pixelwaarden van 0-255 naar 0.0-1.0
    rotation_range=180,         # willekeurige rotatie tot 10 graden
    brightness_range=[0.2, 1.8],  # simuleer verschillende belichtingen
    zoom_range=0.4,
    shear_range=0.3,           # simuleer verschillende afstanden
    width_shift_range=0.2,     # kleine horizontale verschuiving
    height_shift_range=0.2,    # kleine verticale verschuiving
    channel_shift_range=50    # kleine verandering in kanalen (kleur)
)

# Validatiedata alleen normaliseren, geen augmentatie
val_gen = ImageDataGenerator(rescale=1./255)

# --- Data inladen ---
# class_mode="binary" omdat we maar 2 klassen hebben: rood en zwart
train_data = train_gen.flow_from_directory(
    os.path.join(SPLIT_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"        # <-- was "categorical", nu "binary" voor 2 klassen
)

val_data = val_gen.flow_from_directory(
    os.path.join(SPLIT_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"        # <-- zelfde aanpassing
)



print(f"Klassen gevonden: {train_data.class_indices}")  # toont {'rood': 0, 'zwart': 1}

# --- Model opbouw ---
# MobileNetV2 is voorgetraind op ImageNet (miljoenen afbeeldingen)
# We bevriezen de basis zodat alleen onze nieuwe lagen worden getraind
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False,
                         weights="imagenet")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# base_model.trainable = False  # voorgetrainde lagen niet aanpassen

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(4, activation="softmax")
])

# binary_crossentropy is de juiste verliesfunctie voor 2 klassen
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss="categorical_crossentropy",  # <-- was "binary_crossentropy"
              metrics=["accuracy"])

# --- EarlyStopping ---
# Stopt automatisch als het model niet meer verbetert na 5 epochs
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# --- Training starten ---
print("\nStart training op CPU...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=60,
    callbacks=[early_stop],
    verbose=1
)

# --- Visualisatie van de trainingsprogressie ---
acc        = history.history['accuracy']
val_acc    = history.history['val_accuracy']
loss       = history.history['loss']
val_loss   = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

# Grafiek 1: Accuracy per epoch
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc,     label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validatie Accuracy')
plt.title('Progressie: Accuracy')
plt.legend(loc='lower right')

# Grafiek 2: Loss per epoch
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss,     label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validatie Loss')
plt.title('Progressie: Loss')
plt.legend(loc='upper right')

plt.show()

test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    os.path.join(SPLIT_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss:     {test_loss:.4f}")


# --- Model opslaan ---
model.save(os.path.join(BASE_DIR, "model_symbols_filtered.keras"))
print("\nModel opgeslagen als model_symbols_filtered.keras")
