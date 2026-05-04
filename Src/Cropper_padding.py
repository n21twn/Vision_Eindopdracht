import os
import cv2
import numpy as np
import yaml
import random

# Paden
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_PATH   = os.path.join(BASE_DIR, "Dataset2", "data.yaml") #De YAML met class names
IMAGE_DIR   = os.path.join(BASE_DIR, "Dataset2", "train", "images") # De originele foto's
LABEL_DIR   = os.path.join(BASE_DIR, "Dataset2", "train", "labels") # De YOLO-annotaties (zelfde naam als foto, maar .txt)
OUTPUT_DIR  = os.path.join(BASE_DIR, "object_crops_2") # Waar de gecropte objecten en achtergrond-crops worden opgeslagen.

TARGET_SIZE = 224
BG_PER_IMG  = 3  # Hoeveel achtergrond-crops per originele foto

# functie om class names te laden uit de YAML (voor submappen in output)
def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

# functie om de plaatje te croppen volgens de YOLO-annotatie, en daarna te resizen met padding zodat het een vierkant van 224x224 wordt.
def crop_and_padding(image, yolo_bbox, target_size=224, is_bg=False):
    h_img, w_img, _ = image.shape
    _, x_ptr, y_ptr, w_ptr, h_ptr = yolo_bbox
    
    w = int(w_ptr * w_img)
    h = int(h_ptr * h_img)
    x_center = int(x_ptr * w_img)
    y_center = int(y_ptr * h_img)

    x1 = max(0, x_center - (w // 2))
    y1 = max(0, y_center - (h // 2))
    x2 = min(w_img, x1 + w)
    y2 = min(h_img, y1 + h)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0: return None

    scale = target_size / max(cropped.shape[0], cropped.shape[1])
    new_w, new_h = int(cropped.shape[1] * scale), int(cropped.shape[0] * scale)
    resized = cv2.resize(cropped, (new_w, new_h))

    # Voor objecten: random kleur. Voor background: vaker zwart of grijs padding
    c = random.randint(0, 255) if not is_bg else 0
    square = np.full((target_size, target_size, 3), [c, c, c], dtype=np.uint8)

    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    square[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return square

# Functie om te checken of een nieuwe background crop overlapt met bestaande objecten (om te voorkomen dat we per ongeluk een kaart croppen als achtergrond)
def is_overlapping(new_box, existing_boxes, threshold=0.1):
    """Check of de nieuwe background box een bestaande kaart raakt."""
    bx, by, bw, bh = new_box
    for ex_box in existing_boxes:
        ex_x, ex_y, ex_w, ex_h = ex_box
        
        # Bereken overlap (Intersection)
        x_left = max(bx - bw/2, ex_x - ex_w/2)
        y_top = max(by - bh/2, ex_y - ex_h/2)
        x_right = min(bx + bw/2, ex_x + ex_w/2)
        y_bottom = min(by + bh/2, ex_y + ex_h/2)

        if x_right > x_left and y_bottom > y_top:
            intersection = (x_right - x_left) * (y_bottom - y_top)
            # Als de overlap te groot is, weiger de crop
            if intersection > 0: 
                return True
    return False

# In je process_dataset loop gebruik je dit dan:

def process_dataset(image_dir, label_dir, output_dir, class_names):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None: continue

        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        existing_boxes = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    data = list(map(float, line.split()))
                    existing_boxes.append(data[1:]) # bewaar coords voor BG check
                    
                    class_name = class_names[int(data[0])].replace(" ", "_")
                    final_img = crop_and_padding(image, data, TARGET_SIZE)
                    
                    if final_img is not None:
                        out_path = os.path.join(output_dir, class_name)
                        os.makedirs(out_path, exist_ok=True)
                        cv2.imwrite(os.path.join(out_path, f"{img_name}_{i}.jpg"), final_img)

        # Genereer Background crops op random plekken
        for b in range(BG_PER_IMG):
            bx, by = random.random() * 0.8 + 0.1, random.random() * 0.8 + 0.1
            bw, bh = 0.2, 0.2
    
            if not is_overlapping([bx, by, bw, bh], existing_boxes):
                bg_img = crop_and_padding(image, [0, bx, by, bw, bh], TARGET_SIZE, is_bg=True)
                bg_path = os.path.join(output_dir, "background")
                os.makedirs(bg_path, exist_ok=True)
                cv2.imwrite(os.path.join(bg_path, f"bg_{img_name}_{b}.jpg"), bg_img)
                # opslaan in background map...
        # for b in range(BG_PER_IMG):
        #     # Simpel: pak een random vierkant en hoop dat er geen kaart zit 
        #     # (In een dataset met weinig kaarten werkt dit prima)
        #     bw, bh = 0.15, 0.15
        #     bx, by = random.random() * 0.8 + 0.1, random.random() * 0.8 + 0.1
        #     bg_data = [0, bx, by, bw, bh]
            
        #     bg_img = crop_and_padding(image, bg_data, TARGET_SIZE, is_bg=True)
        #     if bg_img is not None:
        #         bg_path = os.path.join(output_dir, "background")
        #         os.makedirs(bg_path, exist_ok=True)
        #         cv2.imwrite(os.path.join(bg_path, f"bg_{img_name}_{b}.jpg"), bg_img)

    print("Klaar met croppen inclusief background!")

if __name__ == "__main__":
    names = load_class_names(YAML_PATH)
    process_dataset(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR, names)