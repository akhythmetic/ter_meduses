\# Métriques drone et étalonnage



\## Source : email de Quentin (10/05/2026)



\## Hauteurs de vol



| Session    | Hauteur | Statut     |

|------------|---------|------------|

| 2025-06-06 | Variable | À étalonner par vidéo |

| 2025-06-18 | Variable | À étalonner par vidéo |

| 2025-06-30 | 25 m fixe | Étalonner 1 fois pour la session |

| 2025-08-08 | 25 m fixe | Idem |

| 2025-08-15 | 25 m fixe | Idem |



\## Références de taille connues



| Objet | Dimension | Notes |

|---|---|---|

| Flotteur triangulaire | 75 mm de côté | Sur certaines vidéos avec méduses marquées |

| Sphère du flotteur | 45 mm de diamètre | Idem |

| Bassine noire | 530 mm de diamètre | Visible sur certaines vidéos |

| Barque (largeur) | 1.90 m | Quand visible |

| Barque (longueur) | 5.50 m | Quand visible — étalon le plus fiable |



\## Méthode d'étalonnage



GSD = taille\_réelle (m) / taille\_pixel (px)



Toutes les mesures pixel issues de YOLO/tracker doivent être 

multipliées par GSD pour obtenir la valeur en mètres.



\## TODO

\- \[ ] Étalonner session 2025-06-30 (vidéo de référence DJI\_0013)

&#x20;     avec la barque ou la bassine si présentes

\- \[ ] Étalonner sessions 2025-08-08 et 2025-08-15 (25m fixes)

\- \[ ] Étalonner vidéo par vidéo pour 2025-06-06 et 2025-06-18

\- \[ ] Demander à Quentin les valeurs exactes des bassines/barques

&#x20;     mentionnées (au cas où il y aurait plusieurs barques 

&#x20;     différentes)

