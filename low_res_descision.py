# U ovom kodu odredjujemo koja rezolucija je bolja za analizu

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from osum import read_raw, im_norm

# Putanja do foldera
folder_path = 'DugeAnatomije_NSpine_HeraAT'
files = [f for f in os.listdir(folder_path) if f.lower().endswith('.fxd')]

# Inicijalizacija CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

for file_name in files: 
    full_path = os.path.join(folder_path, file_name)
    rezultat = read_raw(full_path, crop_flag=1)
    
    if rezultat:
        im = rezultat[0].astype(np.float32)
        
        # 1. Osnovna transformacija (Log + CLAHE)
        im_log = im_norm(np.log1p(im))
        im_8bit = (im_log * 255).astype(np.uint8)
        im_preprocessed = clahe.apply(im_8bit)
        
        # 2. Promena rezolucije na dimenzije
        # Koristimo INTER_AREA za smanjivanje slika
        im_128 = cv2.resize(im_preprocessed, (128, 128), interpolation=cv2.INTER_AREA)
        im_190 = cv2.resize(im_preprocessed, (190, 190), interpolation=cv2.INTER_AREA)
        
        # 3. Prikaz za poredjenje
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Poređenje rezolucija za fajl: {file_name}")
        
        ax[0].imshow(im_128, cmap='gray')
        ax[0].set_title("Rezolucija 128x128")
        ax[0].axis('on') # broj piksela
        
        ax[1].imshow(im_190, cmap='gray')
        ax[1].set_title("Rezolucija 190x190")
        ax[1].axis('on')
        
        plt.tight_layout()
        plt.show()