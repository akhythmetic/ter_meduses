# Résumé Phase 6 — Comparaison triple méthode de mesure de pulsation
## TER Méduses — M1 MIASHS | DJI_0013.MP4

---

## Top 5 chiffres clés

| # | Chiffre | Signification |
|---|---------|---------------|
| 1 | **90.9 %** couverture YOLOseg | Le segmenteur rate 9.1 % des détections — toutes sur bbox < 300 px² — ce qui justifie d'avoir 3 méthodes plutôt qu'une seule |
| 2 | **0.547** corrélation Pearson bbox / SAM | SAM produit une information non redondante avec la bbox (≠ simple mise à l'échelle), ce qui valide methodologiquement l'ajout de SAM |
| 3 | **0.921 vs 0.293** amplitude relative moyenne YOLOseg vs SAM | YOLOseg amplifie le signal de pulsation 3× plus que SAM ; utile pour détecter les pics, mais aussi plus bruité (CV moyen 0.29 vs 0.09) |
| 4 | **~0.23 Hz** fréquence dominante (3 méthodes convergentes) | Légèrement en dessous de la bande biologique *Rhizostoma pulmo* (0.3–1.5 Hz), probablement dû à des oscillations lentes du drone superposées au signal de pulsation |
| 5 | **5 balisées simultanées sur 35s** (133–163s) | Fenêtre rare permettant une comparaison inter-individuelle directe — contexte du segment démo |

---

## Livrables produits

### Vidéos / Données brutes
| Fichier | Description |
|---------|-------------|
| `results/trajectories_detection_DJI0013.csv` | Tracking ByteTrack + bbox area (11 061 détections, ~2.7 Mo) |
| `results/trajectories_seg_yolovseg_DJI0013.csv` | Même clés + mask area YOLOv8-seg (90.9 % non-NaN) |
| `results/trajectories_seg_sam_DJI0013.csv` | Même clés + mask area SAM 2.1-b (100 % non-NaN) |
| `results/DJI0013_three_signals_demo.mp4` | Vidéo démo 1080p/29.97fps, segment 133–163s, 20.8 Mo |

### Analyse
| Fichier | Description |
|---------|-------------|
| `results/pulsation_methods_comparison.csv` | Tableau récap 5 balisées × 3 méthodes (amplitude, CV, freq peaks, freq FFT) |
| `results/pulsation_analysis_report.md` | Rapport complet avec tableaux, résultats clés, limites, recommandation |
| `results/balisees_seg_coverage.csv` | Couverture YOLOseg par balisée physique |
| `logs/phase6_seg_coverage.csv` | Taux de NaN YOLOseg par bin de bbox_area |

---

## Figures — liste et justification

### `figures/pulsation_three_signals.png`
**Grille 5 lignes × 3 colonnes** : signal temporel brut (médiane mobile k=3) de chaque balisée
pour les 3 méthodes, avec les pics détectés marqués (×).

> *Pourquoi cette figure ?* C'est la visualisation de base qui confirme que les 3 signaux oscillent
> en phase et permet de voir visuellement si les pics détectés sont plausibles biologiquement.
> B3 (référence, 100 % coverage) sert de comparateur.

---

### `figures/pulsation_amplitude_comparison.png`
**Boîte à moustaches** (3 boîtes = 3 méthodes) de l'amplitude relative P95–P5 / médiane,
avec les p-values du test de Wilcoxon apparié.

> *Pourquoi cette figure ?* L'amplitude relative est le seul indicateur de sensibilité
> comparable entre méthodes indépendamment de l'échelle physique. Le test Wilcoxon évalue
> si les différences d'amplitude observées sont statistiquement significatives sur n=5 balisées
> (appariement naturel : même individu, 3 mesures).

---

### `figures/pulsation_frequency_comparison.png`
**Barres groupées** (5 balisées × 3 méthodes) des fréquences FFT dominantes, avec
une bande grisée matérialisant la plage biologique attendue [0.3–1.5 Hz].

> *Pourquoi cette figure ?* La fréquence est la grandeur physiologique d'intérêt premier
> (rythme cardiaque-analogue de la méduse). La bande biologique permet de juger si les méthodes
> détectent bien le signal de pulsation ou des artefacts. La convergence des 3 méthodes
> valide la robustesse de l'estimation de fréquence.

---

### `figures/pulsation_correlation_matrix.png`
**Heatmap 3×3** des corrélations de Pearson centrées-réduites par balisée,
entre les 3 signaux (bbox_area, mask_yolovseg, mask_sam).

> *Pourquoi cette figure ?* C'est la figure de validation méthodologique centrale :
> si les 3 méthodes étaient redondantes, toutes les corrélations seraient proches de 1.
> La corrélation bbox/SAM = 0.547 montre que SAM capture de l'information complémentaire
> (morphologie du manteau vs taille de la boîte englobante). La corrélation YOLOseg/SAM
> évalue si les deux masques convergent sur la forme réelle.

---

## Limites principales

1. **Fréquence sous la bande biologique** : les ~0.23 Hz observés suggèrent que le signal
   de pulsation est mélangé à des oscillations lentes (mouvement drone, courant).
   Sans stabilisation IMU ou soustraction du fond, la FFT ne peut pas isoler proprement
   la composante de pulsation.

2. **Résolution temporelle** : `vid_stride=6` → 4.995 Hz effectif → Nyquist = 2.5 Hz.
   Suffisant pour *Rhizostoma pulmo* (max ~1.5 Hz) mais pas pour détecter d'éventuels
   artefacts à haute fréquence.

3. **YOLOv8-seg mAP50-95 mask = 0.139** : les masques seg sont approximatifs sur les méduses
   < 30 px de diamètre. Cela gonfle l'amplitude relative de YOLOseg artificiellement.

4. **n=5 balisées** : les tests statistiques (Wilcoxon) ont une puissance faible avec n=5.
   Les conclusions doivent rester qualitatives.

5. **Dataset intra-vidéo** : tous les résultats sont valables pour DJI_0013 uniquement.
   La généralisation à d'autres conditions (lumière, turbidité, profondeur) n'a pas été testée.

---

## Recommandation méthode

| Priorité | Méthode | Cas d'usage |
|----------|---------|-------------|
| 1 | **SAM 2.1-b** | Analyse fine de morphologie, GPU disponible, couverture 100 %, CV minimal |
| 2 | **Bbox YOLOv8n** | Contrainte de ressources, fréquence uniquement (pas de forme) |
| 3 | **YOLOv8-seg** | Après ré-entraînement sur dataset plus grand (mAP50-95 mask cible ≥ 0.5) |

---

*Généré automatiquement — Phase 6.4 | `src/analyze_pulsation_comparison.py` + `src/render_demo_video.py` + `src/inference_triple_video.py`*
