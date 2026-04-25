import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_PATH   = os.path.join(BASE_DIR, "Dataset2", "train", "images",
                          "000990204_jpg.rf.8084f3679ab659e2de1c442cb8debdad.jpg")
MODEL_PATH = os.path.join(BASE_DIR, "model_met_filters.keras")

# model en foto laden
model   = load_model(MODEL_PATH)
img     = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# klassenamen ophalen uit de split map
train_dir    = os.path.join(BASE_DIR, "object_crops_split", "train")
klasse_namen = sorted(os.listdir(train_dir))

# zelfde preprocessing als tijdens training
def canny_combinatie(img):
    img_uint8  = img.astype(np.uint8)
    gray       = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    canny      = cv2.Canny(gray, threshold1=50, threshold2=150)
    canny_3k   = np.stack([canny, canny, canny], axis=-1)
    origineel  = img / 255.0
    canny_norm = canny_3k.astype(np.float32) / 255.0
    return np.clip(origineel + 0.5 * canny_norm, 0, 1)

# configuratie
VENSTER_GROOTTES = [80]  # meerdere groottes
STAP             = 20                   # kleine stap voor betere dekking
CONFIDENCE       = 0.98

detections = []
h, w = img_rgb.shape[:2]

for venster_grootte in VENSTER_GROOTTES:
    print(f"Zoeken met venster {venster_grootte}x{venster_grootte}...")
    
    for y in range(0, h - venster_grootte, STAP):
        for x in range(0, w - venster_grootte, STAP):
            # venster uitknippen
            venster = img_rgb[y:y+venster_grootte, x:x+venster_grootte]
            
            # opschalen naar 224x224 zodat model het herkent
            venster_224 = cv2.resize(venster, (224, 224))
            
            # zelfde preprocessing als training
            venster_processed = canny_combinatie(venster_224.astype(np.float32))
            venster_input     = np.expand_dims(venster_processed, axis=0)
            
            # door het model
            voorspelling = model.predict(venster_input, verbose=0)
            confidence   = np.max(voorspelling)
            klasse_id    = np.argmax(voorspelling)
            
            if confidence > CONFIDENCE:
                detections.append((x, y, venster_grootte, klasse_id, confidence))

print(f"\nGevonden detections: {len(detections)}")
for x, y, grootte, klasse_id, conf in detections:
    naam = klasse_namen[klasse_id]
    print(f"  {naam} met {conf:.2f} zekerheid op ({x}, {y}) grootte {grootte}")

def nms(detections, overlap_drempel=0.5):
    """Non-Maximum Suppression — verwijdert overlappende boxes en houdt alleen de sterkste"""
    if len(detections) == 0:
        return []

    # sorteren op confidence van hoog naar laag
    detections = sorted(detections, key=lambda d: d[4], reverse=True)
    bewaard    = []

    while detections:
        beste = detections.pop(0)
        bewaard.append(beste)

        overig = []
        for det in detections:
            x1, y1, g1, _, _ = beste
            x2, y2, g2, _, _ = det

            # bereken overlap tussen twee boxes
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + g1, x2 + g2)
            iy2 = min(y1 + g1, y2 + g2)

            overlap_w = max(0, ix2 - ix1)
            overlap_h = max(0, iy2 - iy1)
            overlap   = overlap_w * overlap_h

            opp1 = g1 * g1
            opp2 = g2 * g2
            iou  = overlap / (opp1 + opp2 - overlap)

            # alleen bewaren als overlap klein genoeg is
            if iou < overlap_drempel:
                overig.append(det)

        detections = overig

    return bewaard

# nms toepassen
detections_nms = nms(detections)
print(f"\nNa NMS: {len(detections_nms)} detections")

# bounding boxes tekenen op de foto
img_result = img_rgb.copy()
for x, y, grootte, klasse_id, conf in detections_nms:
    naam = klasse_namen[klasse_id]
    cv2.rectangle(img_result,
                  (x, y),
                  (x + grootte, y + grootte),
                  (255, 0, 0), 2)
    cv2.putText(img_result, f"{naam} {conf:.2f}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(img_result)
plt.axis("off")
plt.title("Detections na NMS")
plt.show()