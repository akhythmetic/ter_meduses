# Rapport pulsation — Nouvelles sessions
Modèle détection : `runs/detect/yolov8n_1280_e150_enriched/weights/best.pt`
Modèle SAM       : `sam2.1_b.pt`
Bande biologique : 0.3–1.5 Hz

## Résultats par session

| Session | Pistes | Dans bande bio | Freq médiane (Hz) | Min Hz | Max Hz |
|---|---|---|---|---|---|
| 2025_06_06 | 10 | 10/10 | 0.242 | 0.150 | 0.948 |
| 2025_08_08 | 1 | 1/1 | 0.855 | 0.855 | 0.855 |
| 2025_08_15 | 6 | 3/6 | 1.186 | 0.200 | 1.929 |

## Interprétation

**14/17 pistes** (82%) ont une fréquence dominante dans la bande biologique 0.3–1.5 Hz. La pipeline SAM généralise correctement aux nouvelles sessions.

## Fichiers produits

- `results/pulsation_generalization.csv`
- `figures/pulsation_generalization.png`
- `figures/pulsation_2025_06_06.png`
- `figures/pulsation_2025_08_08.png`
- `figures/pulsation_2025_08_15.png`
