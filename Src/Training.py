import os
import tensorflow as tf
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


# paden naar de gesplitte dataset
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(BASE_DIR, "object_crops_split")
 
# MobileNetV2 verwacht 224x224 plaatjes
IMG_SIZE   = 224
BATCH_SIZE = 32
 
# pixel waardes normaliseren van 0-255 naar 0-1, dit werkt beter voor neurale netwerken
train_gen = ImageDataGenerator(rescale=1./255)
val_gen   = ImageDataGenerator(rescale=1./255)
 
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
    epochs=10,
    verbose=1
)
 