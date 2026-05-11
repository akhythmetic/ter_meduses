"""
Visualisation des trajectoires — DJI_0013.

Produit 3 figures dans results/figures/ :
  1. trajectories_overview.png  — toutes les trajectoires (balisées en couleur)
  2. trajectories_balisees.png  — balisées seules, recentrées (drift retiré)
  3. trajectories_timeseries.png — x(t) et y(t) de chaque balisée

Usage :
  python src/plot_trajectories.py
  python src/plot_trajectories.py --show   # affiche en plus des sauvegardes
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "results" / "trajectories_DJI0013_final.csv"
FIGURES_DIR = ROOT / "results" / "figures"

BALISEE_COLORS = {
    "Balisée #1": "#e6194b",
    "Balisée #2": "#3cb44b",
    "Balisée #3": "#4363d8",
    "Balisée #4": "#f58231",
    "Balisée #5": "#911eb4",
}
FPS = 5


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df["time_s"] = df["frame_idx"] / FPS
    return df.sort_values(["jelly_id", "frame_idx"])


# ── Figure 1 : vue d'ensemble ────────────────────────────────────────────────
def plot_overview(df: pd.DataFrame, ax: plt.Axes) -> None:
    non_b = df[df["marked"] == 0]
    for jid, sub in non_b.groupby("jelly_id"):
        ax.plot(sub["x_center"], sub["y_center"], color="#cccccc", lw=0.5, alpha=0.5)

    for balisee, color in BALISEE_COLORS.items():
        sub = df[df["balisee_id"] == balisee]
        ax.plot(sub["x_center"], sub["y_center"], color=color, lw=1.5,
                label=balisee, zorder=3)
        # Start point
        first = sub.iloc[0]
        ax.scatter(first["x_center"], first["y_center"], color=color, s=60,
                   marker="o", zorder=4)

    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title("Trajectoires — DJI_0013 (gris=non balisées)")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_aspect("equal")


# ── Figure 2 : balisées recentrées (drift retiré) ───────────────────────────
def plot_balisees_recentered(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Recentre chaque balisée sur son point de départ pour comparer les mouvements relatifs."""
    for balisee, color in BALISEE_COLORS.items():
        sub = df[df["balisee_id"] == balisee].copy()
        if sub.empty:
            continue
        x0, y0 = sub.iloc[0]["x_center"], sub.iloc[0]["y_center"]
        ax.plot(sub["x_center"] - x0, sub["y_center"] - y0,
                color=color, lw=1.5, label=balisee)
        ax.scatter(0, 0, color=color, s=60, marker="o", zorder=4)

    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Δx depuis départ (pixels)")
    ax.set_ylabel("Δy depuis départ (pixels)")
    ax.set_title("Trajectoires balisées — recentrées (dérive relative)")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_aspect("equal")


# ── Figure 3 : séries temporelles ───────────────────────────────────────────
def plot_timeseries(df: pd.DataFrame, axes) -> None:
    ax_x, ax_y = axes
    for balisee, color in BALISEE_COLORS.items():
        sub = df[df["balisee_id"] == balisee]
        if sub.empty:
            continue
        ax_x.plot(sub["time_s"], sub["x_center"], color=color, lw=1, label=balisee)
        ax_y.plot(sub["time_s"], sub["y_center"], color=color, lw=1)

    ax_x.set_ylabel("x (pixels)")
    ax_x.set_title("Position x des balisées dans le temps")
    ax_x.legend(fontsize=7)
    ax_y.set_xlabel("Temps (s)")
    ax_y.set_ylabel("y (pixels)")
    ax_y.set_title("Position y des balisées dans le temps")


def main(show: bool = False) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    plot_overview(df, ax1)
    fig1.tight_layout()
    fig1.savefig(FIGURES_DIR / "trajectories_overview.png", dpi=150)
    print(f"Sauvegardé : {FIGURES_DIR / 'trajectories_overview.png'}")

    # Figure 2
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plot_balisees_recentered(df, ax2)
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "trajectories_balisees.png", dpi=150)
    print(f"Sauvegardé : {FIGURES_DIR / 'trajectories_balisees.png'}")

    # Figure 3
    fig3, axes3 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    plot_timeseries(df, axes3)
    fig3.tight_layout()
    fig3.savefig(FIGURES_DIR / "trajectories_timeseries.png", dpi=150)
    print(f"Sauvegardé : {FIGURES_DIR / 'trajectories_timeseries.png'}")

    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(show=args.show)
