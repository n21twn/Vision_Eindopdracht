# Splitfolder is een library die dataverzamelingen automatisch kan splitsen in train/val/test mappen, met behoud van de mappenstructuur (klassen).
import splitfolders
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root
INPUT_DIR  = os.path.join(BASE_DIR, "object_crops")
OUTPUT_DIR = os.path.join(BASE_DIR, "object_crops_split")


# 70% train, 20% validatie, 10% test
RATIO = (0.7, 0.2, 0.1)
SEED  = 1  # Zelfde seed = zelfde split, voor als na gemaakt moet worden.

# block for als van uit ander bestand wordt gerund.
if __name__ == "__main__":
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Ratio:  {RATIO[0]*100:.0f}% train / {RATIO[1]*100:.0f}% val / {RATIO[2]*100:.0f}% test")

# splitfolders.ratio() maakt de splitsing en kopieert de bestanden naar de output map.
    splitfolders.ratio(
        input=INPUT_DIR,
        output=OUTPUT_DIR,
        seed=SEED,
        ratio=RATIO,
        move=False  # Kopieer, verplaats niet — origineel blijft intact
    )

    # Tel hoeveel klassen en bestanden er zijn
    train_dir = os.path.join(OUTPUT_DIR, "train")
    classes   = os.listdir(train_dir)
    total     = sum(len(os.listdir(os.path.join(train_dir, c))) for c in classes)

    print(f"\nKlaar!")
    print(f"Klassen: {len(classes)}")
    print(f"Trainplaatjes: {total}")