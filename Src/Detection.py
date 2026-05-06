import cv2
import numpy as np
import tensorflow as tf
import os

# ------------------------------------------------------------------ #
#  SETUP — paden en model inladen                                      #
# ------------------------------------------------------------------ #

# Zoek automatisch de root map van het project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pad naar het getrainde model en de traindata (voor klassenamen)
MODEL_PATH = os.path.join(BASE_DIR, "model_numbers_filtered.keras")
TRAIN_DIR  = os.path.join(BASE_DIR, "object_crops_split_3", "train")

# Model inladen
print("Model wordt geladen...")
model = tf.keras.models.load_model(MODEL_PATH)

# Klassenamen ophalen uit de trainmap (gesorteerde mapnamen = klassenamen)
# Bijvoorbeeld: ['10C', '10D', ..., 'QS', 'background']
class_names = sorted(os.listdir(TRAIN_DIR))


# ------------------------------------------------------------------ #
#  STAP 1 — Kaart vinden in de foto                                   #
#                                                                      #
#  Een speelkaart is wit. We zoeken naar witte pixels in de foto      #
#  en groeperen die samen om de kaart te vinden.                      #
# ------------------------------------------------------------------ #
def apply_filter(image):
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph  = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    morph  = cv2.morphologyEx(morph, cv2.MORPH_OPEN,  kernel)

    return cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)


def find_card_bbox(img):
    """
    Zoekt de speelkaart in de foto via een wit-kleurmasker.
    Geeft de bounding box (x, y, breedte, hoogte) terug, of None als niet gevonden.
    """

    # Schaal de foto naar een vaste hoogte van 600px voor snellere verwerking
    # (minder pixels = sneller, en detectie werkt op elke schaal hetzelfde)
    target_h = 600
    scale    = target_h / img.shape[0]
    small    = cv2.resize(img, (int(img.shape[1] * scale), target_h))
    sh, sw   = small.shape[:2]

    # Zet de afbeelding om van BGR naar HSV kleurruimte
    # HSV maakt het makkelijker om op kleur te filteren dan BGR
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # Maak een masker van alle witte pixels
    # Wit in HSV = lage saturatie (S < 80) en hoge helderheid (V > 130)
    white_mask = cv2.inRange(hsv,
                             np.array([0,   0,   130]),   # minimum HSV
                             np.array([180, 80,  255]))   # maximum HSV

    # Verwijder kleine witte vlekjes (ruis) met een morphological open operatie
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    # Zoek alle witte gebieden (contours) in het masker
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # Geen witte gebieden gevonden

    # Filter kleine contours weg — de kaart moet minimaal 0.5% van het beeld zijn
    min_area = sh * sw * 0.005
    big_cnts = [c for c in contours if cv2.contourArea(c) > min_area]
    if not big_cnts:
        return None

    # Combineer alle grote witte vlekken tot één geheel via convex hull
    # Dit werkt ook als de kaart gedeeltelijk in schaduw ligt en in stukken is gesneden
    all_pts = np.vstack(big_cnts)
    hull    = cv2.convexHull(all_pts)

    # Bereken de rechthoek die de hull omsluit
    x, y, w, h = cv2.boundingRect(hull)

    # Controleer of de verhouding lijkt op een speelkaart (~0.63 breed/hoog)
    ratio = min(w, h) / max(w, h)
    if not (0.45 < ratio < 0.85):
        return None  # Verhouding klopt niet, waarschijnlijk geen kaart

    # Schaal de coördinaten terug naar de originele afbeeldingsgrootte
    inv = 1.0 / scale
    return (int(x * inv), int(y * inv), int(w * inv), int(h * inv)), hull, scale


# ------------------------------------------------------------------ #
#  STAP 2 — Kaart uitsnijden en rechtop zetten                        #
# ------------------------------------------------------------------ #

def extract_card(img, bbox_result):
    (x, y, w, h), _, _ = bbox_result

    margin = 10
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)

    card = img[y1:y2, x1:x2]

    # Zorg dat kaart staand is
    if card.shape[1] > card.shape[0]:
        card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)

    return card  # geen wit-score meer, geen 180° draai


# ------------------------------------------------------------------ #
#  STAP 3 — Hoek van de kaart uitsnijden                              #
#                                                                      #
#  De classifier is getraind op de hoek van de kaart (klein stukje    #
#  met het cijfer/letter en het symbool). We snijden de beste hoek    #
#  uit en schalen die naar 224x224 voor het model.                    #
# ------------------------------------------------------------------ #

def crop_corner(card_img, target_size=224):
    """
    Kiest de beste hoek van de kaart (meeste wit, minste kleur)
    en schaalt die naar target_size x target_size pixels.
    """
    h, w = card_img.shape[:2]

    # Kaart altijd staand maken
    if w > h:
        card_img = cv2.rotate(card_img, cv2.ROTATE_90_CLOCKWISE)
        h, w = card_img.shape[:2]

    # Bereken de grootte van een hoek (30% van de kaart)
    cw = int(w * 0.30)
    ch = int(h * 0.30)

    # Snijd alle 4 hoeken uit
    candidates = {
        "TL": card_img[0:ch,   0:cw  ],   # top-links
        "TR": card_img[0:ch,   w-cw:w],   # top-rechts
        "BL": card_img[h-ch:h, 0:cw  ],   # onder-links
        "BR": card_img[h-ch:h, w-cw:w],   # onder-rechts
    }

    def corner_score(crop):
        """
        Bereken hoe waarschijnlijk het is dat dit de kaarthoek is.
        Kaarthoek = veel wit (achtergrond) + weinig kleur (geen kaartfiguur).
        Score = witte pixels minus gekleurde pixels.
        """
        hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        wit   = cv2.inRange(hsv, np.array([0,  0,  180]), np.array([180, 50,  255]))
        kleur = cv2.inRange(hsv, np.array([0,  80, 80 ]), np.array([180, 255, 255]))
        return np.sum(wit > 0) - np.sum(kleur > 0)

    # Bereken de score voor elke hoek en kies de beste
    scores = {k: corner_score(v) for k, v in candidates.items()}
    best_k = max(scores, key=scores.get)
    print(f"Hoek scores: { {k: int(v) for k, v in scores.items()} } → kies {best_k}")

    # Schaal de gekozen hoek naar 224x224 (zelfde als traindata)
    # corner = candidates[best_k]
    corner = card_img[0:ch, 0:cw]
    return apply_filter(cv2.resize(corner, (target_size, target_size)))

# ------------------------------------------------------------------ #
#  HOOFD-FUNCTIE — voer de volledige detectie uit op één foto         #
# ------------------------------------------------------------------ #

def classify_card(image_path):
    """Laadt een foto, vindt de kaart, en laat het model een voorspelling doen."""

    # Foto inladen
    img = cv2.imread(image_path)
    if img is None:
        print("Bestand niet gevonden.")
        return

    # --- Stap 1: kaart lokaliseren ---
    result = find_card_bbox(img)

    if result is not None:
        print("Kaart gevonden!")
        card_img = extract_card(img, result)
    else:
        # Fallback: kaart niet gevonden, gebruik het midden van de foto
        print("Kaart niet gevonden — fallback naar center-crop.")
        h, w     = img.shape[:2]
        margin_x = w // 6
        margin_y = h // 6
        card_img = img[margin_y:h - margin_y, margin_x:w - margin_x]
        if card_img.shape[1] > card_img.shape[0]:
            card_img = cv2.rotate(card_img, cv2.ROTATE_90_CLOCKWISE)

    # --- Stap 2: hoek uitsnijden ---
    corner_crop = crop_corner(card_img)

    # --- Stap 3: voorspelling maken ---
    # Zet BGR om naar RGB (model is getraind op RGB)
    best_label  = None
    best_conf   = 0
    best_corner = corner_crop

    for rot in [None, cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:

        rotated  = cv2.rotate(corner_crop, rot) if rot is not None else corner_crop.copy()
        filtered = apply_filter(rotated)
        inp      = np.expand_dims(
                    cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB).astype('float32') / 255.0, 0)
        preds    = model.predict(inp, verbose=0)[0]
        idx      = np.argmax(preds)
        conf     = preds[idx]
        print(f"  Rotatie {rot}: {class_names[idx]} ({conf*100:.1f}%)")

        if conf > best_conf:
            best_conf   = conf
            best_label  = class_names[idx]
            best_corner = filtered

    label       = best_label
    conf        = best_conf
    corner_crop = best_corner

    print("-" * 30)
    print(f"MODEL VOORSPELLING: {label}")
    print(f"ZEKERHEID:          {conf * 100:.2f}%")
    print("-" * 30)

    # --- Debug: toon 3 vensters voor visuele controle ---
    def show_resized(title, image, max_h=600):
        """Toon een afbeelding op maximaal max_h pixels hoog."""
        h, w  = image.shape[:2]
        scale = min(1.0, max_h / h)
        cv2.imshow(title, cv2.resize(image, (int(w * scale), int(h * scale))))

    # Venster 1: originele foto met gevonden kaart aangegeven
    vis = img.copy()
    if result is not None:
        (x, y, w, h), hull, scale = result
        hull_orig = (hull / scale).astype(np.int32)
        cv2.drawContours(vis, [hull_orig], -1, (0, 255, 0), 3)  # groene hull
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)  # rode bbox
    show_resized("1. Gevonden kaart", vis)

    # Venster 2: uitgesneden en rechtopgezette kaart
    show_resized("2. Kaart crop", card_img)

    # Venster 3: de hoek-crop die het model als input krijgt
    debug_corner = cv2.resize(corner_crop.copy(), (300, 300))
    cv2.putText(debug_corner, f"{label} {conf * 100:.0f}%", (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("3. Input voor de AI (hoek-crop)", debug_corner)

    # Wacht op toetsdruk voordat de vensters sluiten
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Programma starten ---
classify_card(os.path.join(BASE_DIR, "object_crops_split_3", "test", "Three", "038542874_jpg.rf.4681a67a93686d26874fb71295e2ed33.jpg_2.jpg"))
