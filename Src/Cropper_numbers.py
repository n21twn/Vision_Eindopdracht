import os
import cv2
import yaml

# --- Paden instellen ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_PATH  = os.path.join(BASE_DIR, "Dataset2", "data.yaml")
IMAGE_DIR  = os.path.join(BASE_DIR, "Dataset2", "train", "images")
LABEL_DIR  = os.path.join(BASE_DIR, "Dataset2", "train", "labels")
OUTPUT_DIR = os.path.join(BASE_DIR, "object_crops_numbers_filtered")  # aparte map voor symbool data

TARGET_SIZE = 224  # MobileNetV2 verwacht 224x224 pixels

# --- Mapping van klasse naar rood of zwart ---
# Harten (H) en Ruiten (D) zijn rood
# Schoppen (S) en Klaveren (C) zijn zwart
TWO = ['2D','2S','2H','2C']
THREE = ['3D','3S','3H','3C']
FOUR = ['4D','4S','4H','4C']
FIVE = ['5D','5S','5H','5C']
SIX = ['6D','6S','6H','6C']
SEVEN = ['7D','7S','7H','7C']
EIGHT = ['8D','8S','8H','8C']
NINE = ['9D','9S','9H','9C']
TEN = ['10D','10S','10H','10C']
JACK = ['JD','JS','JH','JC']
QUEEN = ['QD','QS','QH','QC']
KING = ['KD','KS','KH','KC']
ACE = ['AD','AS','AH','AC']




# DIAMENT    = ['AD','2D','3D','4D','5D','6D','7D','8D','9D','10D','JD','QD','KD']
# SPADE      = ['AS','2S','3S','4S','5S','6S','7S','8S','9S','10S','JS','QS','KS']
# HEART      = ['AH','2H','3H','4H','5H','6H','7H','8H','9H','10H','JH','QH','KH']
# CLUB       = ['AC','2C','3C','4C','5C','6C','7C','8C','9C','10C','JC','QC','KC']



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
    if class_name in TWO:
        return "Two"
    elif class_name in THREE:
        return "Three"
    elif class_name in FOUR:
        return "Four"
    elif class_name in FIVE:
        return "Five"
    elif class_name in SIX:
        return "Six"
    elif class_name in SEVEN:
        return "Seven"
    elif class_name in EIGHT:
        return "Eight"
    elif class_name in NINE:
        return "Nine"
    elif class_name in TEN:
        return "Ten"
    elif class_name in JACK:
        return "Jack"
    elif class_name in QUEEN:
        return "Queen"
    elif class_name in KING:
        return "King"
    elif class_name in ACE:
        return "Ace"
    else:
        return None


    
def add_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph  = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    morph  = cv2.morphologyEx(morph, cv2.MORPH_OPEN,  kernel)

    return cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)  # terug naar BGR voor consistente opslag

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
    resized = cv2.resize(cropped, (target_size, target_size))

    return add_filter(resized)  # filter toevoegen voor betere herkenning

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

    tellers = {"Two": 0, "Three": 0, "Four": 0, "Five": 0, "Six": 0, "Seven": 0, "Eight": 0, "Nine": 0, "Ten": 0, "Jack": 0, "Queen": 0, "King": 0, "Ace": 0}
    
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)

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
                    symbol  = kaart_naar_kleur(class_name)
                    # Zet kaartnaam om naar rood of zwart
                    if symbol is None:
                        continue  # onbekende klasse overslaan

                    # Crop en schaal het object
                    final_img = crop_and_resize(image, data, TARGET_SIZE)
                    if final_img is None:
                        continue

                    # Sla op in submap per kleur (rood of zwart)
                    out_path = os.path.join(output_dir, symbol)
                    os.makedirs(out_path, exist_ok=True)
                    cv2.imwrite(os.path.join(out_path, f"{img_name}_{i}.jpg"), final_img)
                    tellers[symbol] += 1

    print(f"Klaar met croppen!")
    for symbol, count in tellers.items():
        print(f"{symbol}: {count} crops")

if __name__ == "__main__":
    names = load_class_names(YAML_PATH)
    process_dataset(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR, names)