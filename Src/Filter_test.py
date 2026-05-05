import cv2
import numpy as np

# ------------------------------------------------------------------ #
#  FILTER TEST — probeer verschillende filters op een crop             #
#  Doel: symbool verduidelijken voor betere herkenning                 #
# ------------------------------------------------------------------ #

def toon_filters(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Kan bestand niet openen: {image_path}")
        return

    img = cv2.resize(img, (224, 224))

    # Filter 1 — Grayscale
    # Kleur doet er niet toe voor symboolherkenning, 1 kanaal ipv 3
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Filter 2 — Vaste threshold
    # Alles boven 127 = wit, eronder = zwart
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Filter 3 — Otsu threshold
    # Automatische drempelwaarde op basis van histogram
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Filter 4 — Canny edge detectie
    # Pakt alleen de randen van het symbool
    edges = cv2.Canny(gray, 50, 150)

    # Filter 5 — Adaptieve threshold
    # Berekent drempel per klein gebied, werkt beter bij ongelijke belichting
    adaptive = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    # Filter 6 — Morphology op Otsu
    # Dicht kleine gaatjes (closing) en verwijder kleine vlekjes (opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph  = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    morph  = cv2.morphologyEx(morph, cv2.MORPH_OPEN,  kernel)

    # Filter 7 — Gaussian blur + Otsu
    # Eerst ruis verwijderen dan threshold, geeft schonere contouren
    blurred      = cv2.GaussianBlur(gray, (5, 5), 0)
    _, blur_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Zet grayscale om naar BGR voor consistente weergave in grid
    def to_bgr(i):
        return cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) if len(i.shape) == 2 else i

    # Grid van 2 rijen x 4 kolommen
    rij1 = np.hstack([to_bgr(img),   to_bgr(gray),     to_bgr(thresh),    to_bgr(otsu)])
    rij2 = np.hstack([to_bgr(edges), to_bgr(adaptive), to_bgr(morph),     to_bgr(blur_otsu)])
    grid = np.vstack([rij1, rij2])

    # Labels toevoegen
    labels = ["Origineel", "Grayscale", "Threshold", "Otsu",
              "Canny",     "Adaptief",  "Morphology","Blur+Otsu"]
    for i, label in enumerate(labels):
        x = (i % 4) * 224 + 5
        y = (i // 4) * 224 + 20
        cv2.putText(grid, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Filter vergelijking", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Loop — test meerdere plaatjes
print("Voer een pad in naar een crop plaatje (of 'stop' om te stoppen)")

while True:
    pad = input("Pad: ").strip().strip('"').strip("'")
    if pad.lower() == "stop":
        break
    toon_filters(pad)