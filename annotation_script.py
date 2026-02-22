# annotation_script.py
# Ručna anotacija parova snimaka duge anatomije
# Parovi: (i,i+1), (i+1,i)
# Varijante: original, rotacija 90° levo (rotL), 90° desno (rotR)

import os
import cv2
import numpy as np
import csv
from osum import read_raw, im_norm

# ===================== PUTANJE =====================
izvorni_folder = 'HeraAT'
ciljni_folder = 'Dataset_HeraAT'
csv_putanja = os.path.join(ciljni_folder, 'anotacije_heraAT.csv')

if not os.path.exists(ciljni_folder):
    os.makedirs(ciljni_folder)

# ===================== CLAHE =====================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ===================== OBRADA + SNIMANJE =====================
def procesiraj_i_snimi(fajl_ime, rot=None):
    """
    rot = None | 'L' | 'R'
    """
    putanja = os.path.join(izvorni_folder, fajl_ime)
    rezultat = read_raw(putanja, crop_flag=1)

    if rezultat is None:
        return None, None

    im = rezultat[0].astype(np.float32)

    # preprocessing
    im_log = im_norm(np.log1p(im))
    im_8bit = (im_log * 255).astype(np.uint8)
    im_clahe = clahe.apply(im_8bit)
    im_190 = cv2.resize(im_clahe, (190, 190), interpolation=cv2.INTER_AREA)

    # rotacije
    sufiks = ''
    if rot == 'L':
        im_190 = cv2.rotate(im_190, cv2.ROTATE_90_COUNTERCLOCKWISE)
        sufiks = '_rotL'
    elif rot == 'R':
        im_190 = cv2.rotate(im_190, cv2.ROTATE_90_CLOCKWISE)
        sufiks = '_rotR'

    base = fajl_ime.replace('.fxd', '').replace('.FXD', '')
    novo_ime = f"{base}{sufiks}.png"
    nova_putanja = os.path.join(ciljni_folder, novo_ime)

    cv2.imwrite(nova_putanja, im_190)

    return novo_ime, im_190


# ===================== ANOTACIJA PARA =====================
def anotiraj_par(imeA, imgA, imeB, imgB, writer):
    separator = np.zeros((190, 5), dtype=np.uint8)
    prikaz = np.hstack((imgA, separator, imgB))
    prikaz_veliki = cv2.resize(prikaz, (800, 400), interpolation=cv2.INTER_NEAREST)

    cv2.imshow(
        '0-Gore | 1-Dole | 2-Levo | 3-Desno | 4-Nije par | ESC-Kraj',
        prikaz_veliki
    )

    print(f"Poredim {imeA}  <->  {imeB}")
    key = cv2.waitKey(0)

    oznaka = -1
    if key == ord('0'):
        oznaka = 0
    elif key == ord('1'):
        oznaka = 1
    elif key == ord('2'):
        oznaka = 2
    elif key == ord('3'):
        oznaka = 3
    elif key == ord('4'):
        oznaka = 4
    elif key == 27:  # ESC
        cv2.destroyAllWindows()
        exit()

    if oznaka != -1:
        writer.writerow([imeA, imeB, oznaka])
        status = "NIJE PAR" if oznaka == 4 else f"STRANA {oznaka}"
        print(f"Zabeleženo: {imeA} – {imeB} -> {status}")


# ===================== GLAVNI PROGRAM =====================
fajlovi = sorted([
    f for f in os.listdir(izvorni_folder)
    if f.lower().endswith('.fxd')
])

# varijante rotacija
rotacije = [
    (None, None),   # original
    ('L', 'L'),     # rotacija ulevo
    ('R', 'R')      # rotacija udesno
]

with open(csv_putanja, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Slika_A', 'Slika_B', 'Oznaka'])

    # sliding window + simetrični parovi
    for i in range(len(fajlovi) - 1):
        f1 = fajlovi[i]
        f2 = fajlovi[i + 1]

        for rotA, rotB in rotacije:

            # (f1, f2)
            imeA, imgA = procesiraj_i_snimi(f1, rotA)
            imeB, imgB = procesiraj_i_snimi(f2, rotB)
            if imgA is not None and imgB is not None:
                anotiraj_par(imeA, imgA, imeB, imgB, writer)

            # (f2, f1)
            imeA, imgA = procesiraj_i_snimi(f2, rotA)
            imeB, imgB = procesiraj_i_snimi(f1, rotB)
            if imgA is not None and imgB is not None:
                anotiraj_par(imeA, imgA, imeB, imgB, writer)

cv2.destroyAllWindows()
print(f"\nAnotacija završena. CSV fajl: {csv_putanja}")