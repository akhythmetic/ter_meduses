import os


LABELS_DIR = r"Chemin du fichier"

old_class = "15" #Ancien numéro de classe
new_class = "0" #Nouveau numéro de classe

count_files = 0
count_boxes = 0

for filename in os.listdir(LABELS_DIR): #Boucle permettant de parcourir tout les fichiers en ne gardant que ceux qui sont au format .txt
    if not filename.endswith(".txt"):
        continue

    path = os.path.join(LABELS_DIR, filename)

    with open(path, "r") as f: #Permet d'ouvrir les fichiers
        lines = f.readlines() #Et de lire son contenu

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: #Ici si la ligne à moins de 5 éléments alors elle est considérée comme invalide et on à la saute
            continue

        if parts[0] == old_class: #Compare si le premier éléments à l'ancien numéro de classe
            parts[0] = new_class    #Si c'est le cas on le modifie par le nouveau
            count_boxes += 1    #Et on effectue une itération sur le count

        new_lines.append(" ".join(parts) + "\n") #Permet de réecrire les lignes au propre avec le nouveau numéro de classe

    with open(path, "w") as f:
        f.writelines(new_lines) #Sert à réecrire le fichier path avec le contenu de new_lines

    count_files += 1    #Et on effectue une itération sur le count

print("Conversion terminée") #Sortie pour voir si cela à bien tout fonctionné
print(f"- Fichiers traités : {count_files}")
print(f"- Boîtes converties : {count_boxes}")
