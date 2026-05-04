import cv2
import numpy as np
import tensorflow as tf
import os



# Zoek automatisch de root map van het project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pad naar het getrainde model en de traindata (voor klassenamen)
MODEL_PATH = os.path.join(BASE_DIR, "model_zonder_filters.keras")
TRAIN_DIR  = os.path.join(BASE_DIR, "object_crops_split_2", "train")

# Model inladen
print("Model wordt geladen...")
model = tf.keras.models.load_model(MODEL_PATH)

# Klassenamen ophalen uit de trainmap (gesorteerde mapnamen = klassenamen)
class_names = sorted(os.listdir(TRAIN_DIR))



#  Een speelkaart is wit. We zoeken naar witte pixels in de foto      #
#  en groeperen die samen om de kaart te vinden.                      #

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
    # min_area = sh * sw * 0.005
    # big_cnts = [c for c in contours if cv2.contourArea(c) > min_area]
    # if not big_cnts:
    #     return None

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


# Kaart uitsnijden en rechtop zetten

def extract_card(img, bbox_result):
    """
    Snijdt de kaart uit de originele foto en zorgt dat hij rechtop staat
    met de kaarthoek (Q♠ symbool) bovenaan.
    """
    (x, y, w, h), _, _ = bbox_result

    # Voeg een kleine rand toe zodat de kaartrand volledig zichtbaar is
    margin = 10
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)

    card = img[y1:y2, x1:x2]

    # Draai de kaart zodat hij altijd staand is (hoogte > breedte)
    if card.shape[1] > card.shape[0]:
        card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)

    h, w = card.shape[:2]

    # Bepaal welke kant boven is door te kijken waar de meeste witte pixels zitten
    # De kaarthoek (met cijfer + symbool) heeft een witte achtergrond
    top_strip = card[0:int(h * 0.20), :]   # bovenste 20% van de kaart
    bot_strip = card[int(h * 0.80):,   :]  # onderste 20% van de kaart

    def white_score(region):
        """Tel het aantal witte pixels in een gebied."""
        hsv  = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        return np.sum(mask > 0)

    top_score = white_score(top_strip)
    bot_score = white_score(bot_strip)
    print(f"Wit-score top={top_score} bot={bot_score}")

    # Als de onderkant witter is staat de kaart ondersteboven → draai 180°
    if bot_score > top_score:
        card = cv2.rotate(card, cv2.ROTATE_180)
        print("Kaart 180° gedraaid")

    return card


#  De classifier is getraind op de hoek van de kaart (klein stukje met het cijfer/letter en het symbool). 
#snijden de beste hoek uit en schalen die naar 224x224 voor het model. 

#!!!!!!!!!!!!!!! Verder aan werken want werkt nog niet
# def crop_corner(card_img, target_size=224):
#     """
#     Kiest de beste hoek van de kaart (meeste wit, minste kleur)
#     en schaalt die naar target_size x target_size pixels.
#     """
#     h, w = card_img.shape[:2]

#     # Kaart altijd staand maken
#     if w > h:
#         card_img = cv2.rotate(card_img, cv2.ROTATE_90_CLOCKWISE)
#         h, w = card_img.shape[:2]

#     # Bereken de grootte van een hoek (30% van de kaart)
#     cw = int(w * 0.30)
#     ch = int(h * 0.30)

#     # Snijd alle 4 hoeken uit
#     candidates = {
#         "TL": card_img[0:ch,   0:cw  ],   # top-links
#         "TR": card_img[0:ch,   w-cw:w],   # top-rechts
#         "BL": card_img[h-ch:h, 0:cw  ],   # onder-links
#         "BR": card_img[h-ch:h, w-cw:w],   # onder-rechts
#     }

#     def corner_score(crop):
#         """
#         Bereken hoe waarschijnlijk het is dat dit de kaarthoek is.
#         Kaarthoek = veel wit (achtergrond) + weinig kleur (geen kaartfiguur).
#         Score = witte pixels minus gekleurde pixels.
#         """
#         hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#         wit   = cv2.inRange(hsv, np.array([0,  0,  180]), np.array([180, 50,  255]))
#         kleur = cv2.inRange(hsv, np.array([0,  80, 80 ]), np.array([180, 255, 255]))
#         return np.sum(wit > 0) - np.sum(kleur > 0)

#     # Bereken de score voor elke hoek en kies de beste
#     scores = {k: corner_score(v) for k, v in candidates.items()}
#     best_k = max(scores, key=scores.get)
#     print(f"Hoek scores: { {k: int(v) for k, v in scores.items()} } → kies {best_k}")

#     # Schaal de gekozen hoek naar 224x224 (zelfde als traindata)
#     corner = candidates[best_k]
#     return cv2.resize(corner, (target_size, target_size))




def classify_card(image_path):
    """Laadt een foto, vindt de kaart, en laat het model een voorspelling doen."""

    # Foto inladen
    img = cv2.imread(image_path)
    if img is None:
        print("Bestand niet gevonden.")
        return

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

    corner_crop = crop_corner(card_img)

    # Zet BGR om naar RGB (model is getraind op RGB)
    img_rgb    = cv2.cvtColor(corner_crop, cv2.COLOR_BGR2RGB)
    # Normaliseer pixelwaarden van 0-255 naar 0.0-1.0
    input_data = img_rgb.astype('float32') / 255.0
    # Voeg een batch-dimensie toe (model verwacht [batch, hoogte, breedte, kanalen])
    input_data = np.expand_dims(input_data, axis=0)

    # Voorspelling uitvoeren
    preds = model.predict(input_data, verbose=0)[0]
    idx   = np.argmax(preds)   # index van de hoogste score
    label = class_names[idx]   # omzetten naar klassenaam
    conf  = preds[idx]         # zekerheid (0.0 - 1.0)

    print("-" * 30)
    print(f"MODEL VOORSPELLING: {label}")
    print(f"ZEKERHEID:          {conf * 100:.2f}%")
    print("-" * 30)

    def show_resized(title, image, max_h=600):
        """Toon een afbeelding op maximaal max_h pixels hoog."""
        h, w  = image.shape[:2]
        scale = min(1.0, max_h / h)
        cv2.imshow(title, cv2.resize(image, (int(w * scale), int(h * scale))))

    vis = img.copy()
    if result is not None:
        (x, y, w, h), hull, scale = result
        hull_orig = (hull / scale).astype(np.int32)
        cv2.drawContours(vis, [hull_orig], -1, (0, 255, 0), 3)  # groene hull
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)  # rode bbox
    show_resized("1. Gevonden kaart", vis)

    
    show_resized("2. Kaart crop", card_img)

    debug_corner = cv2.resize(corner_crop.copy(), (300, 300))
    cv2.putText(debug_corner, f"{label} {conf * 100:.0f}%", (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("3. Input voor de AI (hoek-crop)", debug_corner)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

classify_card(os.path.join(BASE_DIR, "Src", "test.jpeg"))