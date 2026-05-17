# Vision Eindopdracht — Speelkaart Symbool Herkenning

Automatisch herkennen van speelkaartsymbolen (Harten, Ruiten, Schoppen, Klaveren) op foto's met behulp van MobileNetV2 en HSV kleurdetectie.
---
## Vereisten

Zorg dat de volgende packages geïnstalleerd zijn:

```bash
pip install tensorflow 
pip install opencv-python 
pip install pyyaml
pip install splitfolders
pip install numpy
```
---

## STAP 1: Dataset Downloaden
Download de dataset via:
https://universe.roboflow.com/augmented-startups/playing-cards-ow27d/dataset/3

Pak de dataset uit in de root van het project als map `Dataset2/`. De mapstructuur moet er zo uitzien:
---
## Stap 2 — Kaarthoeken uitsnijden
Run het crop script om de hoeken van de kaarten uit de dataset te knippen:

```bash
python src/Cropper_symbols.py
```

Dit genereert de map `object_crops_symbols_filtered/` met per symbool een submap.
---

## Stap 3 — Dataset opsplitsen

Run het splitter script om de data op te splitsen in 70% training, 20% validatie en 10% test:

```bash
python src/Splitter.py
```

Dit genereert de map `object_crops_split_symbols/` met submappen `train/`, `val/` en `test/`.

---

## Stap 4 — Model trainen

> **Let op:** het getrainde model staat al in de repository. Stap 4 kan overgeslagen worden als je het bestaande model wilt gebruiken.

Run het trainingsscript om het model te trainen:

```bash
python src/Training_Symbol.py
```

De training duurt op een CPU tussen de 1 en 2 uur. Het getrainde model wordt opgeslagen als `model_symbols_filtered.keras`.

---

## Stap 5 — Kaart detecteren

Voeg je eigen testfoto's toe aan de map `Img_test/` en pas de onderste regels van `Detection.py` aan:

```python
classify_card(os.path.join(BASE_DIR, "Img_test", "jouw_foto.jpeg"))
```

Run daarna het detectiescript:

```bash
python src/Detection.py
```

Het systeem toont drie vensters:
- **Venster 1** — originele foto met de gevonden kaart aangegeven
- **Venster 2** — uitgesneden kaart
- **Venster 3** — de hoek-crop die het model als input krijgt, met de voorspelling

Druk op een willekeurige toets om naar de volgende foto te gaan.

---