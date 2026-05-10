# Comparaison des trajectoires

- **Référence** : trajectoires_meduses_avec_pixels  (fps=5.0)
- **Nouveau**   : trajectories_DJI0013_yolov8n_1280_stride6  (fps=5.0)

## 1. Statistiques globales

| Métrique | trajectoires_meduses_avec_pixels | trajectories_DJI0013_yolov8n_1280_stride6 | Delta |
| --- | --- | --- | --- |
| Détections totales | 12293 | 11677 | -616 |
| IDs uniques | 18 | 116 | +98 |
| IDs marquées | 5 | 0 | — |
| Frames indexées (min–max) | 0–1200 | 0–1502 | — |

## 2. Longueur des trajectoires (en **frames**)

> ⚠️ Non comparables directement : fps différents entre les deux fichiers.

| Stat | trajectoires_meduses_avec_pixels | trajectories_DJI0013_yolov8n_1280_stride6 |
| --- | --- | --- |
| Moyenne | 682.9 | 100.7 |
| Médiane | 751.5 | 33.0 |
| Max | 1200 | 1037 |

## 3. Longueur des trajectoires (en **secondes** — comparable)

| Stat | trajectoires_meduses_avec_pixels | trajectories_DJI0013_yolov8n_1280_stride6 | Delta |
| --- | --- | --- | --- |
| Moyenne | 136.6s | 20.1s | -116.5s |
| Médiane | 150.3s | 6.6s | -143.7s |
| Max | 240.0s | 207.4s | -32.6s |
| Couverture temporelle moyenne | 137.9s | 20.2s | -117.6s |

## 4. Distribution des longueurs (en secondes)

```
Percentile      Référence      Nouveau
--------------------------------------
p10                 14.5s         0.2s
p25                 57.0s         0.6s
p50                150.3s         6.6s
p75                232.2s        26.0s
p90                239.9s        57.7s
p95                240.0s        70.5s
p100               240.0s       207.4s
```

## 5. IDs individuels — Nouveau CSV

Longueurs de toutes les trajectoires du nouveau fichier :

| jelly_id | frames | durée (s) | marquée |
| --- | --- | --- | --- |
| -1 | 616 | 123.2s | non |
| 1 | 68 | 13.6s | non |
| 2 | 72 | 14.4s | non |
| 3 | 32 | 6.4s | non |
| 4 | 72 | 14.4s | non |
| 5 | 54 | 10.8s | non |
| 6 | 1 | 0.2s | non |
| 7 | 1 | 0.2s | non |
| 19 | 2 | 0.4s | non |
| 21 | 50 | 10.0s | non |
| 33 | 11 | 2.2s | non |
| 38 | 26 | 5.2s | non |
| 44 | 20 | 4.0s | non |
| 280 | 3 | 0.6s | non |
| 301 | 1 | 0.2s | non |
| 471 | 4 | 0.8s | non |
| 475 | 1 | 0.2s | non |
| 627 | 7 | 1.4s | non |
| 642 | 1 | 0.2s | non |
| 714 | 3 | 0.6s | non |
| 715 | 3 | 0.6s | non |
| 716 | 3 | 0.6s | non |
| 717 | 4 | 0.8s | non |
| 723 | 1 | 0.2s | non |
| 764 | 1037 | 207.4s | non |
| 765 | 157 | 31.4s | non |
| 767 | 432 | 86.4s | non |
| 769 | 13 | 2.6s | non |
| 770 | 688 | 137.6s | non |
| 771 | 934 | 186.8s | non |
| 775 | 8 | 1.6s | non |
| 787 | 92 | 18.4s | non |
| 795 | 186 | 37.2s | non |
| 810 | 240 | 48.0s | non |
| 814 | 14 | 2.8s | non |
| 819 | 1 | 0.2s | non |
| 832 | 2 | 0.4s | non |
| 842 | 154 | 30.8s | non |
| 847 | 137 | 27.4s | non |
| 863 | 165 | 33.0s | non |
| 878 | 362 | 72.4s | non |
| 900 | 246 | 49.2s | non |
| 902 | 327 | 65.4s | non |
| 904 | 6 | 1.2s | non |
| 917 | 2 | 0.4s | non |
| 936 | 38 | 7.6s | non |
| 942 | 91 | 18.2s | non |
| 945 | 223 | 44.6s | non |
| 946 | 13 | 2.6s | non |
| 949 | 40 | 8.0s | non |
| 958 | 313 | 62.6s | non |
| 961 | 1 | 0.2s | non |
| 967 | 48 | 9.6s | non |
| 981 | 56 | 11.2s | non |
| 984 | 1 | 0.2s | non |
| 992 | 293 | 58.6s | non |
| 994 | 154 | 30.8s | non |
| 1000 | 1 | 0.2s | non |
| 1002 | 8 | 1.6s | non |
| 1005 | 1 | 0.2s | non |
| 1027 | 2 | 0.4s | non |
| 1030 | 14 | 2.8s | non |
| 1033 | 98 | 19.6s | non |
| 1038 | 82 | 16.4s | non |
| 1047 | 2 | 0.4s | non |
| 1048 | 202 | 40.4s | non |
| 1049 | 2 | 0.4s | non |
| 1052 | 263 | 52.6s | non |
| 1062 | 1 | 0.2s | non |
| 1080 | 7 | 1.4s | non |
| 1088 | 197 | 39.4s | non |
| 1089 | 5 | 1.0s | non |
| 1094 | 44 | 8.8s | non |
| 1113 | 26 | 5.2s | non |
| 1116 | 34 | 6.8s | non |
| 1119 | 26 | 5.2s | non |
| 1124 | 1 | 0.2s | non |
| 1126 | 1 | 0.2s | non |
| 1135 | 43 | 8.6s | non |
| 1136 | 12 | 2.4s | non |
| 1143 | 6 | 1.2s | non |
| 1162 | 1 | 0.2s | non |
| 1165 | 53 | 10.6s | non |
| 1169 | 1 | 0.2s | non |
| 1170 | 75 | 15.0s | non |
| 1178 | 21 | 4.2s | non |
| 1193 | 84 | 16.8s | non |
| 1195 | 73 | 14.6s | non |
| 1199 | 1 | 0.2s | non |
| 1203 | 16 | 3.2s | non |
| 1217 | 349 | 69.8s | non |
| 1219 | 342 | 68.4s | non |
| 1220 | 288 | 57.6s | non |
| 1222 | 86 | 17.2s | non |
| 1224 | 2 | 0.4s | non |
| 1237 | 44 | 8.8s | non |
| 1242 | 1 | 0.2s | non |
| 1246 | 195 | 39.0s | non |
| 1248 | 289 | 57.8s | non |
| 1252 | 206 | 41.2s | non |
| 1257 | 59 | 11.8s | non |
| 1273 | 118 | 23.6s | non |
| 1287 | 3 | 0.6s | non |
| 1288 | 175 | 35.0s | non |
| 1314 | 8 | 1.6s | non |
| 1316 | 93 | 18.6s | non |
| 1318 | 1 | 0.2s | non |
| 1323 | 17 | 3.4s | non |
| 1337 | 1 | 0.2s | non |
| 1344 | 129 | 25.8s | non |
| 1346 | 133 | 26.6s | non |
| 1364 | 5 | 1.0s | non |
| 1366 | 67 | 13.4s | non |
| 1367 | 75 | 15.0s | non |
| 1371 | 56 | 11.2s | non |
| 1399 | 2 | 0.4s | non |

## 6. Note méthodologique

- Les IDs ne sont **pas comparables** entre les deux fichiers : ByteTrack réassigne les IDs indépendamment à chaque session.
- La référence S1 est sous-échantillonnée (~5fps effectifs), le nouveau CSV est à 29.97fps. Les durées en secondes sont donc la bonne unité de comparaison.
- `marked` = 0 dans le nouveau CSV (non rempli automatiquement — à annoter manuellement ou via le script K-means HSV).
