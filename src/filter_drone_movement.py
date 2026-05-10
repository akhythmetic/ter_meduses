"""
Filtre les détections en début de vidéo (mouvement de drone).

Supprime toutes les lignes avec frame_idx < MIN_FRAME_IDX du CSV de
trajectoires et sauvegarde le résultat dans un nouveau fichier.

Usage :
    python src/filter_drone_movement.py \
        --input  results/trajectories_DJI0013_yolov8n_1280_stride6.csv \
        --output results/trajectories_DJI0013_filtered.csv \
        --min-frame 200
"""
import argparse
from pathlib import Path

import pandas as pd


def run(input_path: Path, output_path: Path, min_frame: int) -> None:
    df = pd.read_csv(input_path)

    n_before   = len(df)
    ids_before = df["jelly_id"].nunique()

    df_filtered = df[df["frame_idx"] >= min_frame].copy()

    n_after   = len(df_filtered)
    ids_after = df_filtered["jelly_id"].nunique()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_path, index=False)

    print(f"Seuil applique    : frame_idx >= {min_frame}")
    print(f"Detections avant  : {n_before}  ({ids_before} IDs uniques)")
    print(f"Detections apres  : {n_after}  ({ids_after} IDs uniques)")
    print(f"Supprimees        : {n_before - n_after} detections, {ids_before - ids_after} IDs disparus")
    print(f"Sortie            : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filtre le mouvement de drone en debut de video")
    parser.add_argument("--input",     required=True, type=Path)
    parser.add_argument("--output",    required=True, type=Path)
    parser.add_argument("--min-frame", type=int, default=200,
                        help="Exclure les frames < min-frame (defaut: 200 = 40s a 5fps)")
    args = parser.parse_args()
    run(args.input, args.output, args.min_frame)
