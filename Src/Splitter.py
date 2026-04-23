# Splitfolder is een library die dataverzamelingen automatisch kan splitsen in train/val/test mappen, met behoud van de mappenstructuur (klassen).
import splitfolders
import os
import random
import shutil

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root
INPUT_DIR  = os.path.join(BASE_DIR, "object_crops")
TEMP_DIR   = os.path.join(BASE_DIR, "object_crops_temp")  # tijdelijke map met max plaatjes
OUTPUT_DIR = os.path.join(BASE_DIR, "object_crops_split")


# 70% train, 20% validatie, 10% test
RATIO = (0.7, 0.2, 0.1)
SEED  = 1  # Zelfde seed = zelfde split, voor als na gemaakt moet worden.
MAX_PER_KLASSE = 200



# block for als van uit ander bestand wordt gerund.
if __name__ == "__main__":


    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    random.seed(SEED)

    for klasse in os.listdir(INPUT_DIR):
        klasse_pad = os.path.join(INPUT_DIR, klasse)
        if not os.path.isdir(klasse_pad):
            continue

        plaatjes = os.listdir(klasse_pad)
        random.shuffle(plaatjes)
 
        # maximaal MAX_PER_KLASSE plaatjes nemen
        geselecteerd = plaatjes[:MAX_PER_KLASSE]
 
        # kopieer naar tijdelijke map
        temp_klasse_pad = os.path.join(TEMP_DIR, klasse)
        os.makedirs(temp_klasse_pad)
        for plaatje in geselecteerd:
            shutil.copy(
                os.path.join(klasse_pad, plaatje),
                os.path.join(temp_klasse_pad, plaatje)
            )
 
    print(f"Tijdelijke map aangemaakt met max {MAX_PER_KLASSE} plaatjes per klasse")



    print(f"Input:  {TEMP_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Ratio:  {RATIO[0]*100:.0f}% train / {RATIO[1]*100:.0f}% val / {RATIO[2]*100:.0f}% test")
    
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

# splitfolders.ratio() maakt de splitsing en kopieert de bestanden naar de output map.
    splitfolders.ratio(
        input=TEMP_DIR,
        output=OUTPUT_DIR,
        seed=SEED,
        ratio=RATIO,
        move=False  # Kopieer, verplaats niet — origineel blijft intact
    )

    # Tel hoeveel klassen en bestanden er zijn
    train_dir = os.path.join(OUTPUT_DIR, "train")
    classes   = os.listdir(train_dir)
    total     = sum(len(os.listdir(os.path.join(train_dir, c))) for c in classes)

    shutil.rmtree(TEMP_DIR)  # Opruimen van tijdelijke map
    print(f"Tijdelijke map verwijderd.")

    print(f"\nKlaar!")
    print(f"Klassen: {len(classes)}")
    print(f"Trainplaatjes: {total}")