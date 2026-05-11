# Contexte du projet — Reprise TER Méduses

## Cadre
- TER M1, MIASHS, Université Paul Valéry Montpellier 3
- Encadrants : Sandra Bringay, Jérôme Pasquet
- Commanditaire : Quentin (LIRMM)
- Sujet : Analyse des trajectoires des méduses Rhizostoma pulmo à partir
  de vidéos drone, avec balises (flotteurs jaunes triangulaires reliés
  par fil de 2m à certaines méduses)
- Je reprends le projet en solo (peux déléguer ponctuellement à 1 autre
  personne) après une première phase faite à 4
- 2 semaines avant soutenance qui couvre TOUT le projet (S1 + S2)

## État du travail S1 (déjà fait)
- Analyse exploratoire du CSV méta-données des vidéos (87 Go au total)
- Labellisation manuelle de 1200 frames d'UNE seule vidéo (LabelImg)
  avec sous-échantillonnage 1 frame/6s
- Entraînement YOLOv8n et YOLOv8s sur ces frames (split 80/20 sur la
  même vidéo)
- Tracking pour reconstruire trajectoires, nettoyage manuel des IDs
- Mise en évidence visuelle marquées vs non marquées
- Segmentation K-means en HSV dans bounding boxes pour pulsation
- Export CSV avec ID, taille box, marquée/non, etc.

## Performances YOLOv8n actuelles (à valider en réexécutant)
- mAP50 ≈ 0.715
- Precision ≈ 0.65
- Recall max 57% (à confiance=0)
- Confusion matrix : 814 vrais positifs, 2127 méduses ratées (faux
  négatifs), 23 faux positifs
- Problème principal identifié : RAPPEL faible, méduses très petites
  (0.25%-1.5% de l'image)

## Objectifs validés pour les 2 semaines (par ordre de priorité)
1. PRIORITÉ 1 — Améliorer la détection YOLO (focus sur le rappel)
2. Sortie visuelle pour distinguer méduses balisées (vignette/code couleur)
3. Afficher le temps présent pour chaque méduse sur la vidéo
4. Convertir taille en pixels → taille/distance réelle
   (besoin métriques drone : altitude, focale, taille capteur, GSD)
5. Rapport pulsation / temps de vidéo (ATTENTION : échantillonnage
   actuel à 6s = aliasing pour pulsations de 1-3s — il faudra un
   passage haute fréquence dédié)
6. Généraliser à d'autres vidéos
7. Si temps : analyse de la houle (probablement non atteint)

## Points méthodologiques à anticiper en soutenance
- Le split 80/20 actuel est intra-vidéo → métriques optimistes,
  généralisation pas testée. À assumer ou à corriger en testant sur
  frames d'une 2e vidéo (test out-of-distribution).
- Test de Rayleigh annoncé dans l'état de l'art mais jamais fait.
- Houle annoncée mais jamais faite.
- Interprétation marquées vs non marquées reste très qualitative.

## Pistes techniques à explorer
- SAHI (Slicing Aided Hyper Inference) pour booster le rappel
  sur petits objets
- Ajustement du seuil de confiance (le F1 max est à confiance ~0,
  ce qui est suspect)
- Augmentation de données (luminosité, reflets) pour gérer les
  variations entre vidéos
- Possiblement YOLOv8m ou YOLOv11 si v8s a déjà été testé sans gain

## Ressources
- Colab Pro disponible
- Ordinateur perso assez puissant
- Repo GitHub : ter_meduses (branche à préciser)
- 1200 frames labellisées de la première vidéo
- CSV des vidéos disponibles avec leur qualité

## Ma demande à Claude Code
Commence par explorer le repo, lis les notebooks/scripts existants,
et fais-moi un état des lieux du code (qu'est-ce qui tourne, qu'est-ce
qui est cassé, qu'est-ce qui manque). Ensuite on attaque la priorité 1.

---

## Annotation balisées DJI_0013 (journée 3 — 2026-05-11)

### Résumé
Annotation manuelle des méduses balisées dans `DJI_0013` à partir du
CSV filtré (frame ≥ 200, IDs avec ≥ 10 frames, soit 61 IDs sur 91).
Validation de cohérence spatiale et temporelle réalisée avant remplissage.

### Mapping final validé
| Balisée | IDs | Nb frames |
|---------|-----|-----------|
| Balisée #1 | 767, 936, 946, 958, 1165, 1217 | ~1182 |
| Balisée #2 | 771, 1220, 1371 | ~1262 |
| Balisée #3 | 764 | ~1020 |
| Balisée #4 | 787, 863, 942, 981, 1038, 1089, 1193, 1257, 1344 | ~763 |
| Balisée #5 | 847, 945, 1178, 1237, 1288 | ~600 |

### IDs exclus
- **ID 814** (Balisée #2) : double-détection fantôme de l'ID 771 sur
  14 frames simultanées (= toute sa durée). Exclure évite un doublon
  dans les statistiques.
- **ID 832** (Balisée #5) : 2 frames seulement, y=1611 aberrant
  (reste du groupe entre 952 et 1274). Très probablement une mauvaise
  détection transitoire.
- **ID 1146** : absent du CSV filtré (< 10 frames, éliminé comme bruit).

### Limite identifiée — drift du drone
Les distances inter-IDs apparentes (800–1200 px) ne reflètent PAS
un problème de cohérence mais le mouvement du drone sur ~1300 frames.
Les IDs sont séquentiels (aucun chevauchement significatif).
**Impact pour l'analyse de vitesse** : les coordonnées pixel brutes
ne sont pas dans un référentiel monde fixe. Il faudra soit :
a) utiliser les métadonnées GPS/IMU du drone pour stabiliser, ou
b) travailler uniquement sur la vitesse relative (entre méduses dans
   la même frame), ou
c) limiter l'analyse aux segments courts où le drone est quasi-statique.

### Fichiers produits
- `results/trajectories_DJI0013_final.csv` — CSV avec colonnes
  `marked` (0/1) et `balisee_id` ("Balisée #N" ou vide)
- `src/fill_marked_column.py` — script reproductible de remplissage
- `results/IDs_to_annotate_DJI0013.md` — annotation source mise à jour