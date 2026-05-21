"""
Étape 1 — Sélection des 15 fenêtres de validation humain vs modèle.

Sélectionne 15 fenêtres de 10s dans la trajectoire SAM de DJI_0013 selon
les critères de répartition par balisée, couverture >= 80 %, espacement
et diversité temporelle. Calcule n_pulsations_modele via find_peaks.

Sortie :
  results/windows_selected.csv
  figures/windows_overview.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.ndimage import median_filter

REPO = Path(__file__).resolve().parent.parent

CSV_SAM   = REPO / "results" / "trajectories_seg_sam_DJI0013.csv"
OUT_CSV   = REPO / "results" / "windows_selected.csv"
OUT_FIG   = REPO / "figures" / "windows_overview.png"

FPS_SRC   = 29.97
VID_STRIDE = 6
EFF_FPS   = FPS_SRC / VID_STRIDE       # ~4.995 Hz
WIN_S     = 10.0                        # durée fenêtre (secondes)
WIN_FRAMES = int(WIN_S * FPS_SRC)      # 300 source frames
WIN_PTS   = WIN_FRAMES // VID_STRIDE   # ~50 points CSV par fenêtre
TIME_MIN  = 40.0                        # filtre début instable

BALISEES: dict[str, list[int]] = {
    "Balisee_1": [767, 936, 946, 958, 1165, 1217],
    "Balisee_2": [771, 1220, 1371],
    "Balisee_3": [764],
    "Balisee_4": [787, 863, 942, 981, 1038, 1193, 1257, 1344],
    "Balisee_5": [847, 945, 1178, 1237, 1288],
}

SHORT_NAMES = {
    "Balisee_1": "B1", "Balisee_2": "B2", "Balisee_3": "B3",
    "Balisee_4": "B4", "Balisee_5": "B5",
}

COLORS = {
    "B1": "#e74c3c", "B2": "#f39c12", "B3": "#e040fb",
    "B4": "#00bcd4", "B5": "#4caf50", "controle": "#607d8b",
}

# Objectifs de sélection par balisée
TARGETS: list[tuple[str, int]] = [
    ("Balisee_3", 5),
    ("Balisee_1", 3),
    ("Balisee_2", 2),
    ("Balisee_4", 2),
    ("Balisee_5", 2),
    # 1 contrôle négatif ajouté séparément
]


# ─── Utilitaires ──────────────────────────────────────────────────────────────

def rolling_median(arr: np.ndarray, k: int = 3) -> np.ndarray:
    if len(arr) < k:
        return arr.copy()
    return median_filter(arr.astype(float), size=k, mode="nearest")


def count_peaks(values: np.ndarray, times: np.ndarray) -> tuple[int, float]:
    """Retourne (n_pics, freq_hz) sur le signal SAM, mêmes params que pipeline."""
    valid = np.isfinite(values)
    v = values[valid]; t = times[valid]
    if len(v) < 6:
        return 0, 0.0
    v_sm  = rolling_median(v, k=3)
    med   = np.median(v_sm)
    std   = np.std(v_sm)
    if med == 0 or std == 0:
        return 0, 0.0
    min_dist = max(int(EFF_FPS * 0.5), 1)
    peaks, _ = sp_signal.find_peaks(
        v_sm,
        height=med + 0.3 * std,
        distance=min_dist,
        prominence=max(0.15 * std, 1.0),
    )
    n = len(peaks)
    dur = float(t[-1] - t[0]) if len(t) > 1 else WIN_S
    freq = n / dur if dur > 0 else 0.0
    return n, round(freq, 4)


def get_balisee_series(df: pd.DataFrame, bname: str) -> pd.DataFrame:
    ids = BALISEES[bname]
    sub = df[df["jelly_id"].isin(ids)].copy()
    sub = sub.sort_values("time_s").drop_duplicates("time_s", keep="first")
    return sub.reset_index(drop=True)


def compute_coverage(series: pd.DataFrame, f_start: int, f_end: int) -> float:
    """Fraction de points CSV présents dans [f_start, f_end]."""
    possible = (f_end - f_start) // VID_STRIDE + 1
    present  = series[(series["frame_idx"] >= f_start) &
                      (series["frame_idx"] <= f_end)]
    return len(present) / possible if possible > 0 else 0.0


def window_metrics(series: pd.DataFrame, f_start: int, f_end: int) -> dict:
    sub = series[(series["frame_idx"] >= f_start) &
                 (series["frame_idx"] <= f_end)].copy()
    if len(sub) == 0:
        return {}
    jids     = sorted(sub["jelly_id"].unique().tolist())
    cx       = round(float(sub["x_center"].mean()), 1)
    cy       = round(float(sub["y_center"].mean()), 1)
    area     = round(float(sub["bbox_area_px"].mean()), 1)
    t_start  = round(f_start / FPS_SRC, 3)
    t_end    = round(f_end   / FPS_SRC, 3)
    n_pics, freq = count_peaks(
        sub["mask_area_sam_px"].values,
        sub["time_s"].values,
    )
    return dict(
        frame_start=f_start, frame_end=f_end,
        time_start_s=t_start, time_end_s=t_end,
        duree_s=round(t_end - t_start, 2),
        jelly_ids=jids,
        x_center_moyen=cx, y_center_moyen=cy,
        taille_moyenne_px=area,
        n_pulsations_modele=n_pics,
        freq_modele_hz=freq,
        n_pts=len(sub),
    )


def select_spread(candidates: list[dict], n: int,
                  already_selected: list[dict] | None = None) -> list[dict]:
    """
    Parmi les candidats triés par time_start_s, sélectionne n fenêtres
    maximisant l'espacement temporel (greedy MaxSpread).
    already_selected : fenêtres déjà prises sur la même balisée (évite chevauchement).
    """
    if not candidates:
        return []

    # Exclure les candidats qui chevauchent une fenêtre déjà sélectionnée
    used_ranges: list[tuple[int, int]] = []
    if already_selected:
        used_ranges = [(w["frame_start"], w["frame_end"]) for w in already_selected]

    free = []
    for c in candidates:
        overlap = any(
            not (c["frame_end"] < us or c["frame_start"] > ue)
            for us, ue in used_ranges
        )
        if not overlap:
            free.append(c)

    if not free:
        return []

    free.sort(key=lambda x: x["time_start_s"])
    n = min(n, len(free))
    if n == 1:
        # Prend le milieu de la liste (milieu temporel)
        return [free[len(free) // 2]]

    # Greedy : premier + dernier, puis le point le plus éloigné des déjà pris
    selected = [free[0], free[-1]]
    while len(selected) < n:
        best_c, best_min_dist = None, -1
        for c in free:
            if c in selected:
                continue
            min_d = min(abs(c["time_start_s"] - s["time_start_s"]) for s in selected)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_c = c
        if best_c is None:
            break
        selected.append(best_c)

    selected.sort(key=lambda x: x["time_start_s"])
    return selected


# ─── Sélection principale ──────────────────────────────────────────────────────

def generate_candidates(series: pd.DataFrame, min_cov: float = 0.80,
                        step_s: float = 2.0) -> list[dict]:
    """Génère tous les candidats valides (couverture >= min_cov) pour une balisée."""
    if len(series) == 0:
        return []

    t_min  = series["time_s"].min()
    t_max  = series["time_s"].max()
    f_min  = int(series["frame_idx"].min())
    f_max  = int(series["frame_idx"].max())

    # Pas de 2s en frames source
    step_f = int(step_s * FPS_SRC)

    candidates = []
    f = f_min
    while f + WIN_FRAMES <= f_max + VID_STRIDE:
        f_end = f + WIN_FRAMES - VID_STRIDE
        cov   = compute_coverage(series, f, f_end)
        if cov >= min_cov:
            m = window_metrics(series, f, f_end)
            if m:
                m["coverage"] = round(cov, 3)
                candidates.append(m)
        f += step_f

    return candidates


def find_negative_control(df: pd.DataFrame) -> dict | None:
    """
    Cherche une fenêtre où le tracking d'une balisée s'interrompt au milieu.
    Cible : couverture 30-70 %, présence dans la première moitié > 70 %
    et absence dans la deuxième moitié > 60 %.
    """
    for bname, ids in [("Balisee_4", BALISEES["Balisee_4"]),
                        ("Balisee_5", BALISEES["Balisee_5"]),
                        ("Balisee_1", BALISEES["Balisee_1"])]:
        series = get_balisee_series(df, bname)
        if len(series) == 0:
            continue

        f_min = int(series["frame_idx"].min())
        f_max = int(series["frame_idx"].max())
        step_f = int(2.0 * FPS_SRC)

        f = f_min
        while f + WIN_FRAMES <= f_max + VID_STRIDE:
            f_mid = f + WIN_FRAMES // 2
            f_end = f + WIN_FRAMES - VID_STRIDE

            cov_first  = compute_coverage(series, f,     f_mid)
            cov_second = compute_coverage(series, f_mid, f_end)
            cov_total  = compute_coverage(series, f,     f_end)

            if (cov_first >= 0.65 and cov_second <= 0.35 and
                    0.25 <= cov_total <= 0.70):
                m = window_metrics(series, f, f_end)
                if m:
                    m["coverage"]  = round(cov_total, 3)
                    m["balisee_physique"] = "controle"
                    m["balisee_src"] = bname
                    return m
            f += step_f

    return None


def select_all_windows(df: pd.DataFrame) -> list[dict]:
    all_windows: list[dict] = []
    selected_per_balisee: dict[str, list[dict]] = {b: [] for b in BALISEES}
    global_selected: list[dict] = []

    for bname, n_target in TARGETS:
        short = SHORT_NAMES[bname]
        print(f"\n{bname} ({short}) — objectif {n_target} fenêtres")
        series = get_balisee_series(df, bname)
        candidates = generate_candidates(series, min_cov=0.80, step_s=2.0)
        print(f"  {len(candidates)} candidats valides (cov >= 80 %)")

        chosen = select_spread(candidates, n_target,
                               already_selected=selected_per_balisee[bname])
        selected_per_balisee[bname].extend(chosen)
        global_selected.extend(chosen)

        for w in chosen:
            w["balisee_physique"] = short
            w["balisee_src"]      = bname
        all_windows.extend(chosen)
        print(f"  -> {len(chosen)} sélectionnées : "
              + ", ".join(f"{w['time_start_s']:.0f}s" for w in chosen))

    # Contrôle négatif
    print("\nRecherche contrôle négatif…")
    ctrl = find_negative_control(df)
    if ctrl:
        all_windows.append(ctrl)
        print(f"  -> Contrôle négatif : {ctrl['balisee_src']} à "
              f"{ctrl['time_start_s']:.0f}s (cov={ctrl['coverage']:.0%})")
    else:
        print("  [WARN] Contrôle négatif introuvable — on cherche une fenêtre à couverture partielle quelconque")
        # Fallback : fenêtre avec couverture ~50%
        for bname, ids in BALISEES.items():
            series = get_balisee_series(df, bname)
            cands  = generate_candidates(series, min_cov=0.40, step_s=5.0)
            cands  = [c for c in cands if c.get("coverage", 1) < 0.70]
            if cands:
                w = cands[len(cands) // 2]
                w["balisee_physique"] = "controle"
                w["balisee_src"]      = bname
                all_windows.append(w)
                print(f"  -> Fallback : {bname} à {w['time_start_s']:.0f}s (cov={w.get('coverage',0):.0%})")
                break

    return all_windows


# ─── CSV sortie ───────────────────────────────────────────────────────────────

def build_csv(windows: list[dict]) -> pd.DataFrame:
    # Assigne les IDs W01 à W15
    # Ordre : d'abord par balisée selon TARGETS, puis contrôle
    order = [SHORT_NAMES[b] for b, _ in TARGETS] + ["controle"]
    windows.sort(key=lambda w: (
        order.index(w["balisee_physique"]) if w["balisee_physique"] in order else 99,
        w["time_start_s"],
    ))

    rows = []
    for i, w in enumerate(windows, 1):
        jids = w.get("jelly_ids", [])
        jid_str = str(jids[0]) if len(jids) == 1 else str(jids)
        rows.append({
            "window_id":          f"W{i:02d}",
            "balisee_physique":   w["balisee_physique"],
            "jelly_id":           jid_str,
            "frame_start":        w["frame_start"],
            "frame_end":          w["frame_end"],
            "time_start_s":       w["time_start_s"],
            "time_end_s":         w["time_end_s"],
            "duree_s":            w["duree_s"],
            "x_center_moyen":     w["x_center_moyen"],
            "y_center_moyen":     w["y_center_moyen"],
            "taille_moyenne_px":  w["taille_moyenne_px"],
            "coverage":           w.get("coverage", 1.0),
            "n_pulsations_modele":w["n_pulsations_modele"],
            "freq_modele_hz":     w["freq_modele_hz"],
        })
    return pd.DataFrame(rows)


# ─── Figure overview ──────────────────────────────────────────────────────────

def fig_overview(df: pd.DataFrame, win_df: pd.DataFrame) -> None:
    balisees_order = ["Balisee_5", "Balisee_4", "Balisee_3",
                      "Balisee_2", "Balisee_1"]
    y_labels       = [SHORT_NAMES[b] for b in balisees_order]
    y_pos          = {b: i for i, b in enumerate(balisees_order)}

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle(
        "Vue d'ensemble — 15 fenêtres de validation sur la timeline DJI_0013",
        fontsize=13, fontweight="bold",
    )

    # Fond : trajectoire de chaque balisée (présence)
    for bi, bname in enumerate(balisees_order):
        series = get_balisee_series(df, bname)
        if len(series) == 0:
            continue
        t = series["time_s"].values
        yi = y_pos[bname]
        ax.scatter(t, [yi] * len(t), s=1.5, color="lightgray",
                   alpha=0.5, zorder=1)

    # Fenêtres : rectangles colorés
    height = 0.55
    for _, row in win_df.iterrows():
        bp   = row["balisee_physique"]
        col  = COLORS.get(bp, "#607d8b")
        # Retrouver le balisee_src pour la position Y
        bsrc = None
        for bname, short in SHORT_NAMES.items():
            if short == bp:
                bsrc = bname; break
        if bsrc is None:  # controle
            # Essaie de retrouver la balisée source via le jelly_id
            jid_raw = str(row["jelly_id"]).strip("[]").split(",")[0].strip()
            try:
                jid = int(jid_raw)
                bsrc = next(
                    (b for b, ids in BALISEES.items() if jid in ids),
                    balisees_order[0],
                )
            except ValueError:
                bsrc = balisees_order[0]

        yi = y_pos.get(bsrc, 0)
        rect = mpatches.FancyBboxPatch(
            (row["time_start_s"], yi - height / 2),
            row["duree_s"], height,
            boxstyle="round,pad=0.1",
            linewidth=1.2,
            edgecolor="white",
            facecolor=col,
            alpha=0.85,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            row["time_start_s"] + row["duree_s"] / 2, yi,
            f"{row['window_id']}\n{row['n_pulsations_modele']}p",
            ha="center", va="center", fontsize=7.5,
            fontweight="bold", color="white", zorder=4,
        )

    ax.set_xlim(35, 310)
    ax.set_ylim(-0.8, len(balisees_order) - 0.2)
    ax.set_yticks(range(len(balisees_order)))
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_xlabel("Temps (s)", fontsize=10)
    ax.xaxis.grid(True, alpha=0.3)

    # Légende couleur
    legend_patches = [
        mpatches.Patch(color=COLORS[s], label=s)
        for s in ["B1", "B2", "B3", "B4", "B5", "controle"]
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              ncol=3, fontsize=9, framealpha=0.9)

    ax.set_title(
        f"{len(win_df)} fenêtres × 10s — "
        f"couverture min 80 % — "
        f"n_pulsations = pic détectés par le modèle SAM",
        fontsize=10, loc="left",
    )

    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figure : {OUT_FIG}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Chargement CSV SAM…")
    df = pd.read_csv(CSV_SAM)
    df = df[df["time_s"] >= TIME_MIN].copy()
    print(f"  {len(df)} lignes après filtre time_s >= {TIME_MIN}s")

    print("\n=== Sélection des fenêtres ===")
    windows = select_all_windows(df)

    print(f"\n{len(windows)} fenêtres sélectionnées au total.")
    win_df = build_csv(windows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    win_df.to_csv(OUT_CSV, index=False)
    print(f"[OK] CSV : {OUT_CSV}")

    print("\n=== Résumé ===")
    print(win_df[[
        "window_id", "balisee_physique", "time_start_s", "time_end_s",
        "coverage", "n_pulsations_modele", "freq_modele_hz",
    ]].to_string(index=False))

    print("\nGénération figure overview…")
    fig_overview(df, win_df)


if __name__ == "__main__":
    main()
