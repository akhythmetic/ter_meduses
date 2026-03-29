#code pour éviter de reprendre à zéro le calcul des hogs
import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure

#Listes des dossiers
frames_folder = "Chemin du fichier"    # dossier contenant les frames originales

#Si le fichier est directement importe sur le google collab
#hog_output_folder = "Chemin du fichier"

#Dans le cas ou le fichier hog est ouvert depuis le drive
#hog_output_folder = "Chemin du fichier"

os.makedirs(hog_output_folder, exist_ok=True)

#Regarde et garde les images qui sont de formes .jpg ou .png
image_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.png'))])
existing_hogs = set(os.listdir(hog_output_folder)) #Ici prends la liste du fichier hog_output_folder et le mets dans le set

#Boucle sur les images
for img_file in image_files:
    output_name = f"hog_{img_file}" #Permet de donnée un nom au fichier en rajoutant "hog_" devant le nom de la frame

    #Permet de voir si le calcul de hog sur cette frame est fait, si oui on passe à la prochaine image en disant que cette image est déjà traité
    if output_name in existing_hogs:
        print(f"Déjà traité : {output_name}")
        continue

    img_path = os.path.join(frames_folder, img_file)
    image_bgr = cv2.imread(img_path)

    if image_bgr is None: #Si une image est introuvable ou si elle n'existe pas on passe à la suivante
        print(f"Impossible de lire {img_path}")
        continue

    #Convertir en niveaux de gris
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    #Calcul HOG
    features, hog_image = hog(
        image_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm="L2-Hys"
    )

    #Permet de rendre l'image affichable et puis sauvegardable 
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    #Permet de sauvegarder l'image Hog sans le dossier déjà existant
    output_path = os.path.join(hog_output_folder, output_name)
    cv2.imwrite(output_path, (hog_image_rescaled * 255).astype(np.uint8))

    print(f"HOG sauvegardé pour {img_file}")

print("Calcul des HOG terminé, seuls les fichiers manquants ont été ajoutés.")
