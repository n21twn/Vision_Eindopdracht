**Vision Eindopdracht**

## Dataset
Download de dataset via [https://universe.roboflow.com/augmented-startups/playing-cards-ow27d/dataset/3] en pak uit in de root als `Dataset2/`.
Dan run je `src/cropper.py` om de crops te genereren.
Na het Croppen run het met de Splitter `src/spliter.py` om de data op te splitsen in 70% Training 20% Validatie 10% Test.
Als dit gelukt is kan het bestand `Training.py` gerunt worden om het model te trainen. Dit kan tussen de 1 a 2 uur duren. Voor getrained model staat ook al in de git.
Tot slot run `Detection.py` met getrainede model en voeg onder aan de code bij onderste regel `classify_card(os.path.join(BASE_DIR, "Src", "test.jpeg"))`  Je eigen gekozen plaatje naam in. 
Foto moet toegevoegd worden aan Src diretorie.