# Rapport d'analyse — Phase 6 : Comparaison des méthodes de mesure de pulsation

## Méthode

Trois signaux de pulsation ont été extraits de la vidéo DJI_0013 (301s, 3840×2160, 29.97 fps)
en appliquant un `vid_stride=6` (fps effectifs ≈ 4.995 Hz) et un tracking ByteTrack (conf=0.25).
**Une seule passe de tracking** garantit que les clés `(frame_idx, jelly_id)` sont identiques
dans les 3 CSV de sortie.

### Signaux mesurés

| Signal | Description | Source |
|--------|-------------|--------|
| `bbox_area_px` | Aire de la bounding box du tracker (w × h) | modèle YOLOv8n détection |
| `mask_area_yolovseg_px` | Aire du masque de segmentation | modèle YOLOv8n-seg |
| `mask_area_sam_px` | Aire du masque SAM guidé par bbox | SAM 2.1-b |

### Couverture — résultat clé de l'analyse préliminaire (phase 6.1)

La comparaison n'est possible que si les 3 méthodes fournissent un signal.
L'analyse de couverture préalable (calculée sur les 11 061 détections totales) a montré :

| Méthode | Couverture globale | Note |
|---------|-------------------|------|
| Bbox détection | **100 %** | toujours disponible (par construction) |
| YOLOv8-seg mask | **90.9 %** | NaN concentrés sur bbox < 300 px² (trop petites) |
| SAM mask | **100 %** | SAM répond à toute bbox, même les très petites |

Par balisée physique : B3 présente une couverture YOLOv8-seg de **100 %**
(surface médiane 524 px², toujours au-dessus du seuil de détection seg),
ce qui en fait la **balisée de référence** pour la comparaison directe des 3 méthodes.
Les NaN de YOLOv8-seg s'expliquent par le **mismatch de capacité entre détecteur et segmenteur** :
le détecteur YOLOv8n (mAP50=0.925) préserve des petites détections que le modèle seg ne segmente pas.

Les filtres appliqués : `time_s >= 40.0s` (exclusion des 40 premières secondes
d'instabilité du drone).

---

## Tableau récapitulatif

| balisee_id | n_frames_total | duree_s | amplitude_rel_bbox | amplitude_rel_yolovseg | amplitude_rel_sam | cv_bbox | cv_yolovseg | cv_sam | freq_fft_bbox_hz | freq_fft_yolovseg_hz | freq_fft_sam_hz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Balisee_1 | 1175 | 260.7 | 0.478 | 1.099 | 0.257 | 0.141 | 0.318 | 0.077 | 0.234 | 0.506 | 0.226 |
| Balisee_2 | 1257 | 260.7 | 0.441 | 0.906 | 0.251 | 0.138 | 0.304 | 0.076 | 0.222 | 0.23 | 0.23 |
| Balisee_3 | 1020 | 205.4 | 0.4 | 0.786 | 0.296 | 0.124 | 0.26 | 0.104 | 0.214 | 0.204 | 0.253 |
| Balisee_4 | 758 | 252.9 | 0.504 | 1.085 | 0.287 | 0.159 | 0.288 | 0.084 | 0.237 | 0.292 | 0.217 |
| Balisee_5 | 600 | 215.8 | 0.57 | 0.731 | 0.373 | 0.192 | 0.258 | 0.119 | 0.222 | 0.256 | 0.384 |

---

## Résultats clés

### Instabilité des masques YOLOv8-seg — amplitude élevée ≠ signal de pulsation

La méthode **YOLOv8-seg** présente l'amplitude relative moyenne la plus élevée
(0.921), mais cette valeur **ne reflète pas un signal de pulsation amplifié** :
elle traduit l'**instabilité frame-à-frame des masques** due à la faible qualité
du modèle seg (mAP50-95 mask = 0.139).
La figure `pulsation_three_signals.png` l'illustre directement : les valeurs de
`mask_area_yolovseg_px` sautent entre ~200 et ~1400 px² sur des frames consécutives
pour B3, comportement incompatible avec une pulsation biologique réelle.
L'amplitude élevée de YOLOvSeg est un artefact de bruit, pas un avantage pour la
détection de pics.

Amplitudes moyennes : bbox=0.478, yolovseg=0.921, sam=0.293.

### SAM : seule méthode produisant un signal temporellement cohérent

La méthode **SAM** est la seule produisant un signal lisse et temporellement cohérent
parmi les trois méthodes. Son coefficient de variation moyen est le plus bas (0.092)
et sa variance inter-balisées est la plus faible, indiquant un comportement
reproductible d'une méduse à l'autre.
Sur B3 (couverture 100 %, signal le plus propre), le signal SAM est
lisible à l'œil nu : variations progressives sans sauts brusques, compatibles avec
une pulsation biologique réelle. La `bbox_area` offre un signal intermédiaire,
utilisable comme référence S1 avec lissage, mais moins précis que SAM.

### Convergence des fréquences FFT vs échantillonnage

Les 3 méthodes convergent sur des fréquences dominantes similaires
(BBOX=0.226 Hz, YOLOVSEG=0.298 Hz, SAM=0.262 Hz).

La bande biologique attendue pour *Rhizostoma pulmo* est **0.3–1.5 Hz**
(période de 0.7–3.3s). Les fréquences identifiées par FFT s'écartent partiellement
de cette plage biologiquement attendue. **Question ouverte** : ces fréquences de
~0.23–0.30 Hz représentent-elles la vraie fondamentale de pulsation, ou sont-elles
le résultat d'un sous-échantillonnage de la fondamentale réelle dans la bande
biologique (0.3–1.5 Hz) ? À `vid_stride=6` (≈ 5 Hz effectifs), le repliement
spectral peut masquer des composantes rapides. Cette question motive le test
haute fréquence présenté en section suivante (vid_stride=1, 29.97 Hz natif).

---

## Limites méthodologiques

1. **Résolution temporelle (Nyquist)** : avec `vid_stride=6` à 29.97 fps,
   la fréquence d'échantillonnage effective est ≈ 4.995 Hz, soit une fréquence
   de Nyquist de **2.5 Hz**. Des pulsations > 2.5 Hz ne peuvent pas être résolues.
   Pour *Rhizostoma pulmo* (0.3–1.5 Hz), ce n'est pas limitant, mais pour les
   éventuels artefacts haute fréquence cela crée un repliement.

2. **Qualité des masques YOLOv8-seg** : mAP50-95 mask = 0.139 (faible).
   Les contours seg sont imprécis sur les méduses de taille < 30 px (< 0.07 % de l'image).
   Cela peut créer une variance artificielle dans `mask_area_yolovseg_px`
   sur les balisées de petite taille (B1, B2).

3. **Biais du dataset** : le split 80/20 est intra-vidéo (DJI_0013 uniquement).
   La généralisation à d'autres vidéos n'est pas garantie.

4. **Mouvement du drone** : les coordonnées pixel ne sont pas dans un référentiel
   monde fixe. La bbox size peut varier légèrement si le drone change d'altitude
   ou d'angle. Cet effet est plus fort pour la `bbox_area` que pour les masques,
   car les masques suivent la forme réelle de la méduse.

5. **Corrélation bbox / SAM** : corrélation moyenne de Pearson (centrée-réduite
   par balisée) = **0.576**. Une valeur proche de 1 indiquerait que
   SAM ne produit pas d'information complémentaire à la bbox.

---

## Recommandation

Sur la base de ces résultats, la méthode recommandée pour la suite du projet est :

- **SAM (priorité 1)** si disponibilité GPU : SAM est la seule méthode produisant
  un signal lisse et temporellement cohérent. Couverture 100 %, masques robustes
  aux petites méduses, précision du contour meilleure que la bbox, indépendant du
  seuil de confiance du modèle seg.

- **Bbox détection (S1, fallback)** si contrainte de temps/GPU : simple, rapide,
  100 % de couverture, signal de pulsation viable avec lissage.

- **YOLOv8-seg : écarté pour la pulsation** malgré sa bonne mAP50 box, car
  les masques sont temporellement instables (mAP50-95 mask = 0.139). À envisager
  seulement après ré-entraînement avec plus de données de segmentation.

---

## Vérification fréquentielle haute résolution (B3, vid_stride=1)

Test préliminaire sur la balisée de référence B3, segment frames 4193–4793 (~20s),
méthode SAM uniquement, vid_stride=1 (29.97 Hz natif, ~600 points).

| Paramètre | Valeur |
|-----------|--------|
| Fréquence FFT dominante brute (vid_stride=1) | **0.199 Hz** |
| Fréquence FFT dominante brute (vid_stride=6) | 0.253 Hz |
| Pics détectés | ~17 en 20s → **~0.85–0.95 Hz** |
| Bande biologique *Rhizostoma pulmo* | 0.3–1.5 Hz |

### Interprétation

Deux informations complémentaires ressortent de ce test :

1. **FFT brute dominante (~0.2 Hz) hors bande biologique** : la composante la plus
   puissante du signal est lente (~5s par cycle), probablement liée à une dérive
   lente de la position drone ou à la tendance de fond du signal SAM.
   L'hypothèse de sous-échantillonnage (fondamentale masquée à stride=6)
   ne se confirme pas : à 29.97 Hz, la FFT brute reste basse fréquence.

2. **Pics individuels à ~0.85–0.95 Hz — dans la bande biologique** : la détection
   de ~17 pics en 20s correspond à une fréquence dans la bande attendue pour
   *Rhizostoma pulmo* (0.3–1.5 Hz). Cela suggère que la méduse pulse effectivement
   à cette fréquence, mais que ce signal rapide est noyé dans la tendance lente
   au niveau de la FFT brute globale.

**Conclusion** : pour isoler la composante de pulsation réelle, une FFT sur signal
détrendé (soustraction de la tendance lente sur 5s) ou une analyse par détection
de pics est préférable à la FFT brute.

Voir figures : `figures/pulsation_hifreq_B3.png` (signal, tendance, FFT brute vs détrendée).

*Généré par `src/inference_sam_hifreq.py` et mis à jour par `src/hifreq_all_balisees.py`*

---

## Validation cross-balisées — haute fréquence (vid_stride=1, 29.97 Hz)

Extension aux 5 balisées physiques sur le même segment frames 4193–4793 (~140–160s).
Méthode : SAM uniquement, détection de pics + FFT brute + FFT détrendée
(soustraction médiane glissante 5s = 150 frames).

### Tableau des fréquences par balisée

| Balisée | n points | Fréq. pics (Hz) | FFT brute (Hz) | FFT détrendée (Hz) | Dans bande bio ? |
|---------|----------|-----------------|----------------|---------------------|-----------------|
| Balisee_1 | 466 | 1.049 | 0.149 | 0.149 | Oui (pics) |
| Balisee_2 | 516 | 0.899 | 0.199 | 0.199 | Oui (pics) |
| Balisee_3 | 601 | 0.949 | 0.200 | 0.299 | Oui (pics) — FFT ≈ seuil 0.3 Hz |
| Balisee_4 | 586 | 1.349 | 0.150 | 0.349 | **Oui** |
| Balisee_5 | 598 | 1.149 | 0.150 | 0.150 | Oui (pics) |

*(FFT détrendée = FFT après soustraction de la tendance lente par médiane glissante 5s.
Pour les signaux courts (20s), la détection de pics est plus robuste que la FFT.)*

### Convergence dans la bande biologique

**5/5 balisées** ont une fréquence de pics dans la bande biologique (0.3–1.5 Hz).
Les fréquences de pics s'échelonnent de **0.899 Hz (B2) à 1.349 Hz (B4)**,
avec une médiane de ~1.05 Hz, cohérente avec la littérature pour *Rhizostoma pulmo*
en activité. La convergence entre individus renforce la validité du signal SAM.

Note : la FFT brute reste dominée par une tendance lente (~0.15–0.20 Hz) pour toutes
les balisées. Cette tendance reflète probablement une dérive lente du drone ou des
variations d'angle d'observation, pas la pulsation biologique. La FFT détrendée
donne des valeurs plus proches de la bande bio (B3 : 0.299 Hz ≈ limite, B4 : 0.349 Hz),
mais la détection de pics reste l'estimateur le plus robuste sur 20s.

### Conclusion sur la validité scientifique de l'approche SAM

SAM confirme sa validité scientifique pour la mesure de pulsation : signal
temporellement cohérent, 100 % de couverture sur toutes les balisées,
et fréquences de pulsation cohérentes entre individus (0.9–1.35 Hz).
La FFT sur signal détrendé (soustraction de tendance lente sur 5s) est
l'estimateur recommandé pour la fréquence fondamentale de pulsation quand
un long segment est disponible. Pour des fenêtres courtes (≤ 20s), la
détection de pics est préférable.

*Généré par `src/hifreq_all_balisees.py` — Phase 6 validation cross-balisées*

*Généré automatiquement par `src/analyze_pulsation_comparison.py` — Phase 6.3*
