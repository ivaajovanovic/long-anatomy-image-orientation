# U ovom kodu prolazimo kroz sve snimke u folderu, transformišemo ih i prikazujemo u parovima

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from osum import read_raw, im_pyr_decomp, im_norm

# Putanja do foldera
folder_path = 'HeraAT'

# Sortiramo fajlove da bi parovi (001, 002...) bili jedan do drugog
files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.fxd')])

print(f"Pronađeno {len(files)} fajlova. Prikazujem ih u parovima...")

def obradi_sliku_nivo2(putanja):
    """Pomoćna funkcija koja radi celu transformaciju do Nivoa 2"""
    rezultat = read_raw(putanja, crop_flag=1)
    if rezultat:
        im = rezultat[0].astype(np.float32)
        # 1. Log transformacija
        im_log = im_norm(np.log1p(im))
        # 2. CLAHE (lokalni kontrast)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        im_8bit = (im_log * 255).astype(np.uint8)
        im_clahe = clahe.apply(im_8bit)
        # 3. Gausova piramida - Nivo 2 (1/4 rezolucije)
        LP, GP, Res, size_vec = im_pyr_decomp(im_clahe.astype(np.float32), 3)
        return GP[2]
    return None

# Prolazimo kroz listu sa korakom 2 (i=0, 2, 4...)
for i in range(0, len(files), 2):
    fajl1 = files[i]
    putanja1 = os.path.join(folder_path, fajl1)
    img1 = obradi_sliku_nivo2(putanja1)
    
    img2 = None
    fajl2 = None
    
    # Proveravamo da li postoji sledeći fajl u paru
    if i + 1 < len(files):
        fajl2 = files[i+1]
        putanja2 = os.path.join(folder_path, fajl2)
        img2 = obradi_sliku_nivo2(putanja2)

    # Prikazivanje
    if img1 is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 7)) # Pravimo dva mesta za slike
        
        # Prva slika u paru
        ax[0].imshow(img1, cmap='gray')
        ax[0].set_title(f"Slika A: {fajl1}")
        ax[0].axis('off')
        
        # Druga slika u paru
        if img2 is not None:
            ax[1].imshow(img2, cmap='gray')
            ax[1].set_title(f"Slika B: {fajl2}")
            ax[1].axis('off')
        else:
            ax[1].text(0.5, 0.5, 'Nema para', ha='center', va='center')
            ax[1].axis('off')
            
        plt.tight_layout()
        print(f"Prikazujem par: {fajl1} i {fajl2 if fajl2 else 'NEMA'}")
        plt.show()

print("Kraj pregleda parova.")