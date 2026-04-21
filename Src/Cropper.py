import os
import cv2
import numpy as np
import yaml


# Relatief aan de src/ map waar dit script staat
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root
YAML_PATH   = os.path.join(BASE_DIR, "Dataset2", "data.yaml")
IMAGE_DIR   = os.path.join(BASE_DIR, "Dataset2", "train", "images")
LABEL_DIR   = os.path.join(BASE_DIR, "Dataset2", "train", "labels")
OUTPUT_DIR  = os.path.join(BASE_DIR, "object_crops")


TARGET_SIZE   = 224   # 224px voor Transfer Learning (bijv. MobileNetV2)
PADDING_COLOR = 128   # Middengrijs


def load_class_names(yaml_path):
    """Laad de klassenamen uit data.yaml zodat mappen leesbare namen krijgen
    zoals 'Ace_of_Spades' in plaats van '0', '1', etc."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']  # lijst: index = class_id, waarde = naam


def crop_and_grey_padding(image, yolo_bbox, target_size=224, padding_color=128):
    """Knip één object uit een afbeelding op basis van een YOLO bounding box
    en plaats het gecentreerd in een vierkant met grijze padding.

    YOLO-formaat: class_id  x_center  y_center  width  height  (alles 0-1)
    """
    h_img, w_img, _ = image.shape

    # YOLO geeft alles als percentage van de afbeelding → terugrekenen naar pixels
    _, x_ptr, y_ptr, w_ptr, h_ptr = yolo_bbox
    w        = int(w_ptr * w_img)
    h        = int(h_ptr * h_img)
    x_center = int(x_ptr * w_img)
    y_center = int(y_ptr * h_img)

    # Hoekpunten berekenen
    # Linker/bovenrand: max(0, ...) zodat we nooit buiten het beeld gaan
    # Rechter/onderrand: min(breedte/hoogte, ...) voor hetzelfde aan de andere kant
    x1 = max(0,     x_center - (w // 2))
    y1 = max(0,     y_center - (h // 2))
    x2 = min(w_img, x1 + w)
    y2 = min(h_img, y1 + h)

    cropped = image[y1:y2, x1:x2]

    # Veiligheidscheck: lege crop overslaan
    if cropped.size == 0 or w == 0 or h == 0:
        return None

    # Schalen met behoud van verhoudingen:
    # De langste kant wordt target_size, de andere kant schaalt mee
    scale = target_size / max(x2 - x1, y2 - y1)
    new_w = int((x2 - x1) * scale)
    new_h = int((y2 - y1) * scale)
    resized = cv2.resize(cropped, (new_w, new_h))

    # Vierkant maken: start met grijze achtergrond en plak de crop er gecentreerd in
    square   = np.full((target_size, target_size, 3), padding_color, dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    square[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return square


def process_dataset(image_dir, label_dir, output_dir, class_names):
    """Loop door alle afbeeldingen, lees de YOLO labels,
    knip elk object uit en sla op in een map per klassenaam."""
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Gevonden: {len(image_files)} afbeeldingen in {image_dir}")
    skipped = 0
    saved   = 0

    for img_name in image_files:
        # Zoek het bijbehorende .txt label-bestand
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name) # zet de naam van de afbeelding om naar de naam van het labelbestand

        if not os.path.exists(label_path):
            skipped += 1
            continue

        image = cv2.imread(os.path.join(image_dir, img_name)) # lees de afbeelding in met OpenCV
        if image is None:
            skipped += 1
            continue

        with open(label_path, 'r') as f: # lees het labelbestand
            lines = f.readlines() # elke regel in het labelbestand bevat de informatie over één object in de afbeelding

        for i, line in enumerate(lines): #
            line = line.strip() # verwijder eventuele witruimte aan het begin en einde van de regel
            if not line:
                continue

            data     = list(map(float, line.split())) # splits de regel op spaties en zet elk deel om naar een float (class_id, x_center, y_center, width, height)
            class_id = int(data[0])

            # Klassenaam ophalen uit yaml; val terug op het nummer als het niet bestaat
            class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
            # Spaties vervangen door underscore zodat mapnamen kloppen
            class_name = class_name.replace(" ", "_")

            final_img = crop_and_grey_padding(image, data, TARGET_SIZE, PADDING_COLOR)

            if final_img is not None:
                label_dir_out = os.path.join(output_dir, class_name)
                os.makedirs(label_dir_out, exist_ok=True)

                base_name = os.path.splitext(img_name)[0]
                save_path = os.path.join(label_dir_out, f"{base_name}_{i}.jpg")
                cv2.imwrite(save_path, final_img)
                saved += 1

    print(f"Klaar! {saved} crops opgeslagen in '{output_dir}'")
    print(f"Overgeslagen: {skipped} afbeeldingen (geen label of onleesbaar)")


if __name__ == "__main__":
    class_names = load_class_names(YAML_PATH)
    print(f"Geladen klassen ({len(class_names)}): {class_names}")

    process_dataset(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR, class_names)