import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROPS_DIR  = os.path.join(BASE_DIR, "object_crops")

eerste_map = os.listdir(CROPS_DIR)[0]
eerste_kaart = os.listdir(os.path.join(CROPS_DIR, eerste_map))[0]
img_path = os.path.join(CROPS_DIR, eerste_map, eerste_kaart)


img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mask1 = np.array([[0, -1, 0],
                  [-1, 5, -1],
                  [0, -1, 0]])
sharpening = cv2.filter2D(img_rgb, -1, mask1)

# edge detect. treshold geeft aan hoe snel hij rand pakt 
canny = cv2.Canny(gray, threshold1=1, threshold2=150) 

canny_3k = np.stack([canny, canny, canny], axis=-1) # 3 kanaals maken van de canny randen zodat we die kunnen combineren met de verscherpte afbeelding

# delen door 255 om te normaliseren naar [0, 1] zodat we kunnen combineren zonder dat de waarden te hoog worden
origineel  = img_rgb.astype(np.float32) / 255.0 
canny_norm = canny_3k.astype(np.float32) / 255.0
alpha      = 0.5 # bepalen hoeveel invloed de canny randen hebben in de combinatie, 0 origineel 1 alleen canny randen, 0.5 is een mix van beide
gecombineerd = np.clip(origineel + alpha * canny_norm, 0, 1)



fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(img_rgb);            axes[0].set_title("Origineel")
axes[1].imshow(gray, cmap="gray");  axes[1].set_title("Grijswaarden")
axes[2].imshow(canny, cmap="gray"); axes[2].set_title("Canny randen")
axes[3].imshow(sharpening);         axes[3].set_title("Verscherpte afbeelding")
axes[4].imshow(gecombineerd);       axes[4].set_title("Origineel + Canny")
for ax in axes: ax.axis("off")
plt.tight_layout()
plt.show()