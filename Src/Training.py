import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


# paden naar de gesplitte dataset
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(BASE_DIR, "object_crops_split")
 
IMG_SIZE   = 224
BATCH_SIZE = 32

def canny_combinatie(img):
    # img komt binnen als float32 tussen 0-255
    img_uint8 = img.astype(np.uint8)
    gray      = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    canny     = cv2.Canny(gray, threshold1=50, threshold2=150)
    
    # canny naar 3 kanalen
    canny_3k  = np.stack([canny, canny, canny], axis=-1)
    
    # normaliseren en combineren
    origineel    = img / 255.0
    canny_norm   = canny_3k.astype(np.float32) / 255.0
    gecombineerd = np.clip(origineel + 0.5 * canny_norm, 0, 1)
    
    return gecombineerd


# MobileNetV2 verwacht 224x224 plaatjes


# dan in je ImageDataGenerator:
# pixel waardes normaliseren van 0-255 naar 0-1, dit werkt beter voor neurale netwerken
# preprocessing_function=canny_combinatie
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,       # kleine rotatie want kaarten staan meestal rechtop
    width_shift_range=0.05,  # kleine verschuiving
    height_shift_range=0.05,
    zoom_range=0.05,         # kleine zoom
    horizontal_flip=False    # Voorkom spiegeling want letter J K en Q zijn niet symmetrisch
    )

val_gen = ImageDataGenerator(rescale=1./255)
 
# plaatjes laden vanuit de mappen, keras pakt automatisch de mapnaam als klassenaam
train_data = train_gen.flow_from_directory(
    os.path.join(SPLIT_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

#  zelfde doen voro validatie
val_data = val_gen.flow_from_directory(
    os.path.join(SPLIT_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
 
print(f"Klassen: {train_data.num_classes}")
print(f"Trainplaatjes: {train_data.samples}")
print(f"Valplaatjes: {val_data.samples}")
 
# MobileNetV2 laden zonder de bovenste classificatielagen
# include_top=False zodat we zelf een classificatiekop kunnen toevoegen voor onze 52 klassen
# weights="imagenet" geeft ons de voorgetrainde gewichten van 14 miljoen foto's
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), # 224,224, 3 (RGB)
    include_top=False, # geen classificatielaag, alleen convolutionele basis
    weights="imagenet" # gebruik de voorgetrainde gewichten van ImageNet
)
 
# het basismodel bevriezen zodat de voorgetrainde gewichten niet worden overschreven
# we willen alleen onze eigen lagen trainen
base_model.trainable = False
 
# eigen classificatielagen toevoegen bovenop MobileNetV2
# GlobalAveragePooling2D: https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling2D
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # van 7x7x1280 naar 1280 getallen per plaatje. Geen maxpooling want die geeft 20.000 getallen per plaatje, dat is te veel voor onze kleine dataset.
    Dropout(0.3),              # 30% van neuronen uitzetten om overfitting te voorkomen
    Dense(128, activation="relu"),  # verborgen laag om patronen te combineren
    Dense(train_data.num_classes, activation="softmax")  # uitvoerlaag, 1 per klasse
])
 
 # model compileren met optimizer
model.compile(
    optimizer="adam", 
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
 
print(model.summary())
 
# model trainen, na elke epoch zien we de accuracy op train en validatie data
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    verbose=1
)


model.save(os.path.join(BASE_DIR, "model_met_filters.keras"))
print("Model opgeslagen!")

