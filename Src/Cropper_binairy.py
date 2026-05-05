import os
import cv2
import yaml

# --- Paden instellen ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_PATH  = os.path.join(BASE_DIR, "Dataset2", "data.yaml")
IMAGE_DIR  = os.path.join(BASE_DIR, "Dataset2", "train", "images")
LABEL_DIR  = os.path.join(BASE_DIR, "Dataset2", "train", "labels")
OUTPUT_DIR = os.path.join(BASE_DIR, "object_crops_binairy")  # aparte map voor binaire data

TARGET_SIZE = 224  # MobileNetV2 verwacht 224x224 pixels

# --- Mapping van klasse naar rood of zwart ---
# Harten (H) en Ruiten (D) zijn rood
# Schoppen (S) en Klaveren (C) zijn zwart
ROOD  = ['AH','2H','3H','4H','5H','6H','7H','8H','9H','10H','JH','QH','KH',
         'AD','2D','3D','4D','5D','6D','7D','8D','9D','10D','JD','QD','KD']
ZWART = ['AS','2S','3S','4S','5S','6S','7S','8S','9S','10S','JS','QS','KS',
         'AC','2C','3C','4C','5C','6C','7C','8C','9C','10C','JC','QC','KC']


def load_class_names(yaml_path):
    """Laadt de klassenamen uit het YOLO data.yaml bestand."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']


def kaart_naar_kleur(class_name):
    """
    Zet een kaartnaam om naar 'rood' of 'zwart'.
    Geeft None terug als de kaartnaam niet herkend wordt.
    """
    if class_name in ROOD:
        return "rood"
    elif class_name in ZWART:
        return "zwart"
    else:
        # Onbekende klasse overslaan (bijv. joker of fout label)
        return None


def crop_and_resize(image, yolo_bbox, target_size=224):
    """
    Snijdt één object uit de afbeelding op basis van een YOLO bounding box
    en schaalt het direct naar target_size x target_size pixels.
    YOLO bbox formaat: [class, x_center, y_center, width, height] (genormaliseerd 0-1)
    """
    h_img, w_img, _ = image.shape

    # Zet genormaliseerde YOLO coördinaten om naar absolute pixels
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
        return None

    # Direct stretchen naar 224x224 — geen padding nodig
    return cv2.resize(cropped, (target_size, target_size))


def process_dataset(image_dir, label_dir, output_dir, class_names):
    """
    Verwerkt de hele dataset:
    - Leest elke afbeelding en bijbehorend label bestand
    - Crop elk gelabeld object eruit
    - Zet de kaartnaam om naar 'rood' of 'zwart'
    - Slaat de crop op in een submap per kleur (output/rood/ of output/zwart/)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Haal alle afbeeldingsbestanden op
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    teller_rood  = 0
    teller_zwart = 0

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        image    = cv2.imread(img_path)
        if image is None:
            continue

        # Zoek het bijbehorende label bestand
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    data = list(map(float, line.split()))

                    # Zet class index om naar kaartnaam (bijv. 48 → "QS")
                    class_name = class_names[int(data[0])].replace(" ", "_")

                    # Zet kaartnaam om naar rood of zwart
                    kleur = kaart_naar_kleur(class_name)
                    if kleur is None:
                        continue  # onbekende klasse overslaan

                    # Crop en schaal het object
                    final_img = crop_and_resize(image, data, TARGET_SIZE)
                    if final_img is None:
                        continue

                    # Sla op in submap per kleur (rood of zwart)
                    out_path = os.path.join(output_dir, kleur)
                    os.makedirs(out_path, exist_ok=True)
                    cv2.imwrite(os.path.join(out_path, f"{img_name}_{i}.jpg"), final_img)

                    # Bijhouden hoeveel crops per kleur
                    if kleur == "rood":
                        teller_rood += 1
                    else:
                        teller_zwart += 1

    print(f"Klaar met croppen!")
    print(f"  Rode crops:   {teller_rood}")
    print(f"  Zwarte crops: {teller_zwart}")
    print(f"  Totaal:       {teller_rood + teller_zwart}")


if __name__ == "__main__":
    names = load_class_names(YAML_PATH)
    process_dataset(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR, names)