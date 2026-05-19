"""
Génère figures/pulsation_three_signals_smoothed.png :
grille 5×3 identique à pulsation_three_signals.png mais avec
lissage par médiane glissante fenêtre=5 frames sur les 3 signaux.
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.ndimage import median_filter

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO = Path(__file__).resolve().parent.parent

CSV_DET = REPO / "results" / "trajectories_detection_DJI0013.csv"
CSV_SEG = REPO / "results" / "trajectories_seg_yolovseg_DJI0013.csv"
CSV_SAM = REPO / "results" / "trajectories_seg_sam_DJI0013.csv"
OUT_FIG = REPO / "figures" / "pulsation_three_signals_smoothed.png"

FPS_SOURCE  = 29.97
VID_STRIDE  = 6
EFF_FPS     = FPS_SOURCE / VID_STRIDE
TIME_FILTER = 40.0
SMOOTH_K    = 5   # fenêtre médiane glissante

BALISEES: dict[str, list[int]] = {
    "Balisee_1": [767, 936, 946, 958, 1165, 1217],
    "Balisee_2": [771, 1220, 1371],
    "Balisee_3": [764],
    "Balisee_4": [787, 863, 942, 981, 1038, 1193, 1257, 1344],
    "Balisee_5": [847, 945, 1178, 1237, 1288],
}

BALISEE_NOTES = {
    "Balisee_3": "100% YOLOseg coverage — référence propre",
}

COLORS_3 = {"bbox": "#e74c3c", "yolovseg": "#00bcd4", "sam": "#e040fb"}
LABELS_3  = {"bbox": "Bbox area", "yolovseg": "YOLOv8-seg mask", "sam": "SAM mask"}


def load_and_join() -> pd.DataFrame:
    det = pd.read_csv(CSV_DET)
    seg = pd.read_csv(CSV_SEG)[["frame_idx", "jelly_id",
                                 "mask_area_yolovseg_px",
                                 "mask_perimeter_yolovseg_px",
                                 "matched_iou"]]
    sam = pd.read_csv(CSV_SAM)[["frame_idx", "jelly_id",
                                 "mask_area_sam_px",
                                 "mask_perimeter_sam_px"]]
    df = (det
          .merge(seg, on=["frame_idx", "jelly_id"], how="left")
          .merge(sam, on=["frame_idx", "jelly_id"], how="left"))
    return df[df["time_s"] >= TIME_FILTER].copy()


def get_series(df: pd.DataFrame, bname: str) -> pd.DataFrame:
    ids = BALISEES[bname]
    sub = df[df["jelly_id"].isin(ids)].copy()
    sub = sub.sort_values("time_s").drop_duplicates("time_s", keep="first")
    return sub.reset_index(drop=True)


def smooth(arr: np.ndarray, k: int = SMOOTH_K) -> np.ndarray:
    if len(arr) < k:
        return arr.copy()
    return median_filter(arr.astype(float), size=k, mode="nearest")


def make_figure(df: pd.DataFrame) -> None:
    methods = [
        ("bbox",     "bbox_area_px"),
        ("yolovseg", "mask_area_yolovseg_px"),
        ("sam",      "mask_area_sam_px"),
    ]

    fig, axes = plt.subplots(5, 3, figsize=(18, 14), sharex=False)
    fig.suptitle(
        "Signaux de pulsation lissés — 3 méthodes × 5 balisées physiques\n"
        f"(médiane glissante fenêtre={SMOOTH_K} frames — comparaison équitable)",
        fontsize=13, fontweight="bold", y=0.99,
    )

    for row_i, (bname, _) in enumerate(BALISEES.items()):
        series = get_series(df, bname)
        t      = series["time_s"].values
        note   = BALISEE_NOTES.get(bname, "")
        note_str = f"  [{note}]" if note else ""

        for col_j, (lbl, col) in enumerate(methods):
            ax = axes[row_i][col_j]
            v  = series[col].values.astype(float)
            valid = np.isfinite(v)

            if valid.sum() < 5:
                ax.text(0.5, 0.5, "Données insuffisantes",
                        ha="center", va="center", transform=ax.transAxes,
                        color="gray", fontsize=9)
            else:
                t_v   = t[valid]
                v_sm  = smooth(v[valid])    # médiane glissante k=5

                ax.plot(t_v, v_sm, color=COLORS_3[lbl], linewidth=0.9, alpha=0.85)
                ax.fill_between(t_v, v_sm, alpha=0.15, color=COLORS_3[lbl])

                # Pics sur signal lissé
                mu, std, med = np.mean(v_sm), np.std(v_sm), np.median(v_sm)
                if std > 0 and med > 0:
                    min_dist = max(int(EFF_FPS * 0.5), 1)
                    peaks, _ = sp_signal.find_peaks(
                        v_sm,
                        height=med + 0.3 * std,
                        distance=min_dist,
                        prominence=max(0.15 * std, 1.),
                    )
                    if len(peaks):
                        ax.scatter(t_v[peaks], v_sm[peaks],
                                   color=COLORS_3[lbl], s=12, zorder=5,
                                   marker="v", edgecolors="white", linewidths=0.5)

            if row_i == 0:
                ax.set_title(LABELS_3[lbl], fontsize=11, fontweight="bold",
                             color=COLORS_3[lbl])
            if col_j == 0:
                ax.set_ylabel(f"{bname}{note_str}", fontsize=8)
            if row_i == 4:
                ax.set_xlabel("Temps (s)", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda x, _: f"{x/1000:.1f}k" if x >= 1000 else f"{x:.0f}"
                )
            )

    # Légende commune en bas
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=COLORS_3[lbl], linewidth=2, label=LABELS_3[lbl])
        for lbl in ("bbox", "yolovseg", "sam")
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=10,
        framealpha=0.9,
        title=f"Lissage : médiane glissante fenêtre={SMOOTH_K} frames "
              f"(≈{SMOOTH_K/EFF_FPS:.1f}s)",
        title_fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figure sauvegardée : {OUT_FIG}")


def main() -> None:
    print("Chargement des CSV…")
    df = load_and_join()
    print(f"  {len(df)} lignes après filtre time_s >= {TIME_FILTER}s")
    print("Génération de la figure lissée…")
    make_figure(df)


if __name__ == "__main__":
    main()
