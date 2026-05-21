# Rapport de généralisation — Détection méduses

## Test set : session 2025_08_08 (140 images, jamais vues en entraînement)

### Tableau comparatif

| | Modèle DJI_0013 | Modèle enrichi | Delta |
|---|---|---|---|
| **mAP50 (val DJI_0013)** | 0.960 | 0.957 | -0.003 |
| **mAP50 (test 2025_08_08)** | 0.002 | 0.172 | +0.170 |
| **Precision (test)** | 0.028 | 0.825 | +0.797 |
| **Recall (test)** | 0.042 | 0.196 | +0.154 |
| **Dataset** | 1153 frames, 1 session | 1493 frames, 3 sessions | +340 frames |

### Interprétation

Amelioration significative (> 10 pts mAP50) : le re-entrainement multi-session est indispensable pour generaliser. L'enrichissement du dataset avec de nouvelles sessions est une etape cle de la methode.

**Chute de performance OOD confirmée** : le modèle DJI_0013 seul obtient mAP50=0.002 sur
la session test (vs 0.960 sur son propre val set), confirmant un problème de généralisation
documenté à l'étape 2.

**Impact du ré-entraînement** : le modèle enrichi obtient mAP50=0.172 sur le même test,
soit +0.170 par rapport au modèle initial.

**Maintien sur DJI_0013** : le modèle enrichi maintient mAP50=0.957 sur le val DJI_0013
(-0.003 par rapport au modèle initial), confirmant que la diversification du dataset
n'a pas dégradé les performances sur la session originale.

### Fichiers produits
- `results/generalization_comparison.csv`
- `figures/generalization_comparison.png`
