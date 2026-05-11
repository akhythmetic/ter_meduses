"""
Calcul de la vitesse de chaque méduse à partir des déplacements frame-to-frame.
Unité de base : pixels/seconde (fps=5, stride 6 sur vidéo 30fps).

Sortie :
  results/speed_stats_DJI0013.csv       — stats par ID (mean, median, max, std)
  results/trajectories_DJI0013_with_speed.csv — CSV complet avec colonnes de vitesse

ATTENTION dérive drone :
  Les coordonnées pixel sont dans le référentiel caméra (non stabilisé).
  La vitesse pixel inclut donc le mouvement du drone. Pour une vitesse
  absolue, il faudra stabiliser par GPS/IMU ou calcul relatif inter-méduses.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "results" / "trajectories_DJI0013_final.csv"
OUT_FULL = ROOT / "results" / "trajectories_DJI0013_with_speed.csv"
OUT_STATS = ROOT / "results" / "speed_stats_DJI0013.csv"

FPS = 5  # stride-6 sur 30 fps → 5 fps effectifs


def compute_speed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["jelly_id", "frame_idx"]).copy()
    grp = df.groupby("jelly_id")
    df["dx"] = grp["x_center"].diff()
    df["dy"] = grp["y_center"].diff()
    df["displacement_px"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["speed_px_per_s"] = df["displacement_px"] * FPS
    df["dt_frames"] = grp["frame_idx"].diff()
    return df


def speed_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("jelly_id")
        .agg(
            balisee_id=("balisee_id", "first"),
            marked=("marked", "first"),
            nb_frames=("frame_idx", "count"),
            frame_start=("frame_idx", "min"),
            frame_end=("frame_idx", "max"),
            speed_mean_px_s=("speed_px_per_s", "mean"),
            speed_median_px_s=("speed_px_per_s", "median"),
            speed_max_px_s=("speed_px_per_s", "max"),
            speed_std_px_s=("speed_px_per_s", "std"),
        )
        .reset_index()
        .sort_values("jelly_id")
    )


def print_report(stats: pd.DataFrame) -> None:
    print("\n=== VITESSES — DJI_0013 (px/s, fps=5) ===\n")
    print("Méduses balisées :")
    for _, r in stats[stats["marked"] == 1].sort_values("balisee_id").iterrows():
        print(
            f"  ID {r['jelly_id']:>4d}  {str(r['balisee_id']):<12s}  "
            f"moy={r['speed_mean_px_s']:5.1f}  med={r['speed_median_px_s']:5.1f}  "
            f"max={r['speed_max_px_s']:6.1f}  std={r['speed_std_px_s']:5.1f}"
        )
    nb = stats[stats["marked"] == 0]
    print(f"\nMéduses non balisées ({len(nb)} IDs) :")
    print(f"  Vitesse moyenne (des moyennes) : {nb['speed_mean_px_s'].mean():.1f} px/s")
    print(f"  Vitesse médiane (des médianes) : {nb['speed_median_px_s'].median():.1f} px/s")
    print(f"\nFichiers écrits :\n  {OUT_FULL}\n  {OUT_STATS}")


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df = compute_speed(df)
    stats = speed_stats(df)
    df.to_csv(OUT_FULL, index=False)
    stats.to_csv(OUT_STATS, index=False)
    print_report(stats)


if __name__ == "__main__":
    main()
