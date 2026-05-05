import os
import cv2
import numpy as np
import yaml


BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root van het project
YAML_PATH   = os.path.join(BASE_DIR, "Dataset2", "data.yaml")              # YOLO class namen
IMAGE_DIR   = os.path.join(BASE_DIR, "Dataset2", "train", "images")        # Bronafbeeldingen
LABEL_DIR   = os.path.join(BASE_DIR, "Dataset2", "train", "labels")        # YOLO annotaties (.txt)
OUTPUT_DIR  = os.path.join(BASE_DIR, "object_crops_stretch")                 # Output map per klasse

TARGET_SIZE = 224  # MobileNetV2 verwacht 224x224 pixels

def load_class_names(yaml_path):
    """Laadt de klassenamen uit het YOLO data.yaml bestand (bijv. ['AC', 'AD', 'QS', ...])"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def crop_and_resize(image, yolo_bbox, target_size=224):
    """
    Snijdt één object uit de afbeelding op basis van een YOLO bounding box
    en schaalt het direct naar target_size x target_size pixels.
    https://bboxconverter.readthedocs.io/en/latest/explanation/bounding_box_ultimate_guide.html 
    """
    h_img, w_img, _ = image.shape

    # Zet  YOLO coördinaten om naar pixels
    _, x_ptr, y_ptr, w_ptr, h_ptr = yolo_bbox
    w        = int(w_ptr * w_img)
    h        = int(h_ptr * h_img)
    x_center = int(x_ptr * w_img)
    y_center = int(y_ptr * h_img)

    # Bereken de hoekpunten van de bounding box
    x1 = max(0, x_center - (w // 2))
    y1 = max(0, y_center - (h // 2))
    x2 = min(w_img, x1 + w)
    y2 = min(h_img, y1 + h)

    # Uitsnijden
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return None  # Sla lege crops over

    # Direct stretchen naar 224x224 — geen padding nodig
    return cv2.resize(cropped, (target_size, target_size))

def process_dataset(image_dir, label_dir, output_dir, class_names):
    """
    Verwerkt de hele dataset:
    - Leest elke afbeelding en bijbehorend label bestand
    - Crop elk gelabeld object eruit
    - Slaat de crop op in een submap per klasse (bijv. output/QS/foto_0.jpg)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Haal alle afbeeldingsbestanden op
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        image    = cv2.imread(img_path)
        if image is None:
            continue

        # Zoek het bijbehorende label bestand (zelfde naam, .txt extensie)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    data = list(map(float, line.split()))

                    # Zet class index om naar naam (bijv. 48 → "QS")
                    class_name = class_names[int(data[0])].replace(" ", "_")

                    # Crop en schaal het object
                    final_img = crop_and_resize(image, data, TARGET_SIZE)

                    if final_img is not None:
                        # Sla op in submap per klasse
                        out_path = os.path.join(output_dir, class_name)
                        os.makedirs(out_path, exist_ok=True)
                        cv2.imwrite(os.path.join(out_path, f"{img_name}_{i}.jpg"), final_img)

    print("Klaar met croppen!")

if __name__ == "__main__":
    names = load_class_names(YAML_PATH)
    process_dataset(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR, names)