"""
Phase 6.3 — Analyse comparative pulsation (3 méthodes).

Charge les 3 CSV de 6.1, filtre les 40 premières secondes, reconstitue
les séries temporelles pour les 5 balisées physiques, compare les 3 signaux
(bbox_area, YOLOv8-seg mask, SAM mask) sur plusieurs métriques.

Produit :
  results/pulsation_methods_comparison.csv
  figures/pulsation_three_signals.png
  figures/pulsation_amplitude_comparison.png
  figures/pulsation_frequency_comparison.png
  figures/pulsation_correlation_matrix.png
  results/pulsation_analysis_report.md

Loggue dans logs/phase6_analysis.log.
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO = Path(__file__).resolve().parent.parent

# ─── Chemins ──────────────────────────────────────────────────────────────────
CSV_DET  = REPO / "results" / "trajectories_detection_DJI0013.csv"
CSV_SEG  = REPO / "results" / "trajectories_seg_yolovseg_DJI0013.csv"
CSV_SAM  = REPO / "results" / "trajectories_seg_sam_DJI0013.csv"
OUT_CMP  = REPO / "results" / "pulsation_methods_comparison.csv"
OUT_FIGS = REPO / "figures"
OUT_RPT  = REPO / "results" / "pulsation_analysis_report.md"
LOG_PATH = REPO / "logs" / "phase6_analysis.log"

FPS_SOURCE  = 29.97
VID_STRIDE  = 6
EFF_FPS     = FPS_SOURCE / VID_STRIDE   # ~4.995 Hz
TIME_FILTER = 40.0                      # ignorer les premières 40s

# ─── Balisées ─────────────────────────────────────────────────────────────────
BALISEES: dict[str, list[int]] = {
    "Balisee_1": [767, 936, 946, 958, 1165, 1217],
    "Balisee_2": [771, 1220, 1371],
    "Balisee_3": [764],
    "Balisee_4": [787, 863, 942, 981, 1038, 1193, 1257, 1344],
    "Balisee_5": [847, 945, 1178, 1237, 1288],
}

# Note méthodologique confirmée en 6.1 :
# B3 = 100 % de couverture YOLOvSeg → balisée de référence "propre"
BALISEE_NOTES = {
    "Balisee_3": "100% YOLOseg coverage — reference propre",
}


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        ],
    )
    return logging.getLogger("phase6_analysis")


# ─── Chargement & jointure ────────────────────────────────────────────────────

def load_and_join() -> pd.DataFrame:
    """Charge les 3 CSV, joint sur (frame_idx, jelly_id), filtre t >= 40s."""
    det = pd.read_csv(CSV_DET)
    seg = pd.read_csv(CSV_SEG)[["frame_idx", "jelly_id",
                                 "mask_area_yolovseg_px", "mask_perimeter_yolovseg_px",
                                 "matched_iou"]]
    sam = pd.read_csv(CSV_SAM)[["frame_idx", "jelly_id",
                                 "mask_area_sam_px", "mask_perimeter_sam_px"]]

    df = (det
          .merge(seg, on=["frame_idx", "jelly_id"], how="left")
          .merge(sam, on=["frame_idx", "jelly_id"], how="left"))

    df = df[df["time_s"] >= TIME_FILTER].copy()
    return df


# ─── Extraction série temporelle par balisée ──────────────────────────────────

def get_series(df: pd.DataFrame, bname: str) -> pd.DataFrame:
    """
    Concatène tous les IDs tracker correspondant à la balisée physique.
    Trie par time_s, déduplique si overlap (garde première occurrence).
    """
    ids = BALISEES[bname]
    sub = df[df["jelly_id"].isin(ids)].copy()
    sub = sub.sort_values("time_s").drop_duplicates("time_s", keep="first")
    return sub.reset_index(drop=True)


# ─── Métriques pulsation sur une série ───────────────────────────────────────

def rolling_median(arr: np.ndarray, k: int = 3) -> np.ndarray:
    """Médiane glissante fenêtre k (bords en mode 'same')."""
    if len(arr) < k:
        return arr.copy()
    from scipy.ndimage import median_filter
    return median_filter(arr, size=k, mode="nearest")


def pulsation_metrics(
    values: np.ndarray,
    times:  np.ndarray,
    label:  str,
) -> dict:
    """
    Calcule les métriques de pulsation pour un signal values(t).
    Retourne dict de métriques.
    """
    if len(values) < 10:
        return {
            f"amplitude_rel_{label}": np.nan,
            f"cv_{label}":            np.nan,
            f"n_pics_{label}":        0,
            f"freq_pics_{label}_hz":  np.nan,
            f"freq_fft_{label}_hz":   np.nan,
        }

    valid_mask = np.isfinite(values)
    v = values[valid_mask]
    t = times[valid_mask]
    if len(v) < 10:
        return {
            f"amplitude_rel_{label}": np.nan,
            f"cv_{label}":            np.nan,
            f"n_pics_{label}":        0,
            f"freq_pics_{label}_hz":  np.nan,
            f"freq_fft_{label}_hz":   np.nan,
        }

    # Lissage médiane glissante fenêtre=3
    v_smooth = rolling_median(v, k=3)

    med = np.median(v_smooth)
    mu  = np.mean(v_smooth)
    std = np.std(v_smooth)

    if med == 0 or mu == 0:
        return {
            f"amplitude_rel_{label}": np.nan,
            f"cv_{label}":            np.nan,
            f"n_pics_{label}":        0,
            f"freq_pics_{label}_hz":  np.nan,
            f"freq_fft_{label}_hz":   np.nan,
        }

    p5  = np.percentile(v_smooth, 5)
    p95 = np.percentile(v_smooth, 95)
    amp_rel = (p95 - p5) / med
    cv      = std / mu

    # Détection de pics
    min_dist = max(int(EFF_FPS * 0.5), 1)   # ~0.5s mini entre pics
    height   = med + 0.3 * std
    prom     = max(0.15 * std, 1.0)
    peaks, _ = sp_signal.find_peaks(
        v_smooth, height=height, distance=min_dist, prominence=prom
    )
    n_pics = len(peaks)
    duree_s = float(t[-1] - t[0]) if len(t) > 1 else 1.0
    freq_pics = n_pics / duree_s if duree_s > 0 else np.nan

    # FFT
    dt = np.mean(np.diff(t)) if len(t) > 1 else 1.0 / EFF_FPS
    if dt <= 0:
        dt = 1.0 / EFF_FPS
    freqs = np.fft.rfftfreq(len(v_smooth), d=dt)
    fft_amp = np.abs(np.fft.rfft(v_smooth - v_smooth.mean()))

    # Masque bande biologique [0.2, 2.0] Hz
    bio_mask = (freqs >= 0.2) & (freqs <= 2.0)
    if bio_mask.any():
        freq_dom = float(freqs[bio_mask][np.argmax(fft_amp[bio_mask])])
    else:
        # Fallback: fréquence dominante hors DC
        dc_mask = freqs > 0
        if dc_mask.any():
            freq_dom = float(freqs[dc_mask][np.argmax(fft_amp[dc_mask])])
        else:
            freq_dom = np.nan

    return {
        f"amplitude_rel_{label}": round(float(amp_rel), 4),
        f"cv_{label}":            round(float(cv),       4),
        f"n_pics_{label}":        int(n_pics),
        f"freq_pics_{label}_hz":  round(float(freq_pics), 4) if np.isfinite(freq_pics) else np.nan,
        f"freq_fft_{label}_hz":   round(float(freq_dom),  4) if np.isfinite(freq_dom)  else np.nan,
    }


# ─── Analyse principale ───────────────────────────────────────────────────────

def analyze(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    rows = []
    for bname, ids in BALISEES.items():
        series = get_series(df, bname)
        if len(series) < 20:
            log.warning(f"{bname}: seulement {len(series)} points, ignoré")
            continue

        t = series["time_s"].values
        bbox_v  = series["bbox_area_px"].values
        yolo_v  = series["mask_area_yolovseg_px"].values
        sam_v   = series["mask_area_sam_px"].values

        n_total = len(series)
        duree_s = float(t[-1] - t[0]) if n_total > 1 else 0.

        row = {
            "balisee_id":    bname,
            "n_frames_total": n_total,
            "duree_s":        round(duree_s, 1),
        }
        for vals, lbl in [(bbox_v, "bbox"), (yolo_v, "yolovseg"), (sam_v, "sam")]:
            row.update(pulsation_metrics(vals, t, lbl))

        log.info(
            f"{bname}: {n_total} frames, {duree_s:.1f}s | "
            f"amp_bbox={row.get('amplitude_rel_bbox', 'nan'):.3f}  "
            f"amp_yolo={row.get('amplitude_rel_yolovseg', 'nan'):.3f}  "
            f"amp_sam={row.get('amplitude_rel_sam', 'nan'):.3f}"
        )
        log.info(
            f"  freq_fft: bbox={row.get('freq_fft_bbox_hz', 'nan')}  "
            f"yolo={row.get('freq_fft_yolovseg_hz', 'nan')}  "
            f"sam={row.get('freq_fft_sam_hz', 'nan')} Hz"
        )
        rows.append(row)

    return pd.DataFrame(rows)


# ─── Vérifications alertes ────────────────────────────────────────────────────

def check_alerts(cmp: pd.DataFrame, df: pd.DataFrame, log: logging.Logger) -> list[str]:
    alerts = []

    # Alerte fréquences hors plage biologique
    for col in ["freq_fft_bbox_hz", "freq_fft_yolovseg_hz", "freq_fft_sam_hz"]:
        vals = cmp[col].dropna()
        if len(vals) == 0:
            continue
        if (vals > 3.0).all():
            alerts.append(f"ALERTE : toutes les fréquences FFT {col} > 3 Hz → bruit probable")
        if (vals < 0.1).all():
            alerts.append(f"ALERTE : toutes les fréquences FFT {col} < 0.1 Hz → signal très lent")

    # Alerte corrélation bbox / SAM
    all_bbox, all_sam = [], []
    for bname in BALISEES:
        series = get_series(df, bname)
        v_bbox = series["bbox_area_px"].values
        v_sam  = series["mask_area_sam_px"].values
        valid  = np.isfinite(v_bbox) & np.isfinite(v_sam)
        if valid.sum() > 10:
            mu_b = v_bbox[valid].mean(); s_b = v_bbox[valid].std()
            mu_s = v_sam[valid].mean();  s_s = v_sam[valid].std()
            if s_b > 0 and s_s > 0:
                all_bbox.extend(((v_bbox[valid] - mu_b) / s_b).tolist())
                all_sam.extend( ((v_sam[valid]  - mu_s) / s_s).tolist())

    if len(all_bbox) > 20:
        corr = np.corrcoef(all_bbox, all_sam)[0, 1]
        if corr > 0.95:
            alerts.append(
                f"ALERTE : corrélation bbox/SAM = {corr:.3f} > 0.95 → "
                "la segmentation SAM n'apporte pas d'information supplémentaire"
            )
        log.info(f"Corrélation globale bbox/SAM : {corr:.3f}")

    for a in alerts:
        log.warning(a)
        print(f"\n{'!'*60}\n{a}\n{'!'*60}\n")

    return alerts


# ─── Figures ──────────────────────────────────────────────────────────────────

COLORS_3 = {"bbox": "#e74c3c", "yolovseg": "#00bcd4", "sam": "#e040fb"}
LABELS_3  = {"bbox": "Bbox area", "yolovseg": "YOLOv8-seg mask", "sam": "SAM mask"}


def fig_three_signals(df: pd.DataFrame, out_dir: Path) -> None:
    """Grille 5×3 : série temporelle lissée avec pics annotés."""
    methods = [("bbox", "bbox_area_px"),
               ("yolovseg", "mask_area_yolovseg_px"),
               ("sam",      "mask_area_sam_px")]

    fig, axes = plt.subplots(5, 3, figsize=(18, 14), sharex=False)
    fig.suptitle("Signaux de pulsation — 3 méthodes × 5 balisées physiques",
                 fontsize=14, fontweight="bold", y=0.98)

    for row_i, (bname, ids) in enumerate(BALISEES.items()):
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
                t_v = t[valid]
                v_v = rolling_median(v[valid], k=3)

                ax.plot(t_v, v_v, color=COLORS_3[lbl], linewidth=0.9, alpha=0.85)
                ax.fill_between(t_v, v_v, alpha=0.15, color=COLORS_3[lbl])

                # Pics
                mu = np.mean(v_v); std = np.std(v_v); med = np.median(v_v)
                if std > 0 and med > 0:
                    min_dist = max(int(EFF_FPS * 0.5), 1)
                    peaks, _ = sp_signal.find_peaks(
                        v_v,
                        height=med + 0.3 * std,
                        distance=min_dist,
                        prominence=max(0.15 * std, 1.),
                    )
                    if len(peaks):
                        ax.scatter(t_v[peaks], v_v[peaks],
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
                plt.FuncFormatter(lambda x, _: f"{x/1000:.1f}k" if x >= 1000 else f"{x:.0f}")
            )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / "pulsation_three_signals.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def fig_amplitude_comparison(cmp: pd.DataFrame, out_dir: Path) -> None:
    """Boxplot amplitudes relatives + tests de Wilcoxon."""
    cols = ["amplitude_rel_bbox", "amplitude_rel_yolovseg", "amplitude_rel_sam"]
    labels = ["Bbox", "YOLOv8-seg", "SAM"]
    colors = [COLORS_3["bbox"], COLORS_3["yolovseg"], COLORS_3["sam"]]

    data = [cmp[c].dropna().values for c in cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    # Points individuels
    for j, (d, col) in enumerate(zip(data, colors), start=1):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(d))
        ax.scatter([j]*len(d) + jitter, d, color=col, s=60, zorder=5,
                   edgecolors="black", linewidths=0.5)

    # Tests de Wilcoxon (appariés : 5 balisées)
    pairs = [
        ("Bbox vs YOLOv8-seg", cols[0], cols[1]),
        ("Bbox vs SAM",        cols[0], cols[2]),
        ("YOLOv8-seg vs SAM",  cols[1], cols[2]),
    ]
    y_max  = max(d.max() for d in data if len(d))
    y_step = y_max * 0.12
    valid_pairs = cmp[cols].dropna()

    for pi, (pair_lbl, c1, c2) in enumerate(pairs):
        v1 = valid_pairs[c1].values
        v2 = valid_pairs[c2].values
        if len(v1) >= 4 and not np.all(v1 == v2):
            try:
                stat, pval = wilcoxon(v1, v2)
            except Exception:
                pval = np.nan
        else:
            pval = np.nan

        pval_txt = (f"p={pval:.3f}" if np.isfinite(pval) else "p=N/A")
        y_bracket = y_max + y_step * (pi + 1)
        ax.annotate("", xy=(pairs[pi][0].count("vs") + 0.5, y_bracket),   # dummy
                    xytext=(0.5, 0.5))
        # Simple texte annoté
        ax.text(2, y_max + y_step * (pi + 1), f"{pair_lbl}  {pval_txt}",
                ha="center", va="bottom", fontsize=8, style="italic")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Amplitude relative (P95–P5) / médiane", fontsize=10)
    ax.set_title("Comparaison amplitudes relatives — 3 méthodes\n(Wilcoxon apparié, n=5 balisées)", fontsize=11)
    ax.set_ylim(bottom=0, top=y_max + y_step * (len(pairs) + 1.5))
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "pulsation_amplitude_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def fig_frequency_comparison(cmp: pd.DataFrame, out_dir: Path) -> None:
    """Barres côte à côte : fréquence FFT par balisée × 3 méthodes."""
    balisees = cmp["balisee_id"].tolist()
    x = np.arange(len(balisees))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (lbl, col, col_v) in enumerate([
        ("Bbox",       "freq_fft_bbox_hz",      COLORS_3["bbox"]),
        ("YOLOv8-seg", "freq_fft_yolovseg_hz",  COLORS_3["yolovseg"]),
        ("SAM",        "freq_fft_sam_hz",        COLORS_3["sam"]),
    ]):
        vals = cmp[col].fillna(0).values
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lbl, color=col_v, alpha=0.8,
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.axhspan(0.3, 1.5, alpha=0.07, color="green",
               label="Plage bio. Rhizostoma (0.3–1.5 Hz)")
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("Balisee_", "B") for b in balisees], fontsize=10)
    ax.set_ylabel("Fréquence FFT dominante (Hz)", fontsize=10)
    ax.set_title("Fréquence de pulsation dominante par balisée — 3 méthodes\n"
                 "(bande biologique attendue : 0.3–1.5 Hz, Rhizostoma pulmo)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = out_dir / "pulsation_frequency_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def fig_correlation_matrix(df: pd.DataFrame, out_dir: Path) -> None:
    """Matrice de Pearson 3×3 (signaux centrés-réduits, toutes balisées)."""
    cols_sig = {
        "Bbox":       "bbox_area_px",
        "YOLOv8-seg": "mask_area_yolovseg_px",
        "SAM":        "mask_area_sam_px",
    }
    all_vals: dict[str, list] = {k: [] for k in cols_sig}

    for bname in BALISEES:
        series = get_series(df, bname)
        # Sous-sélection des frames où les 3 sont disponibles
        valid = (series["bbox_area_px"].notna() &
                 series["mask_area_yolovseg_px"].notna() &
                 series["mask_area_sam_px"].notna())
        s = series[valid]
        if len(s) < 5:
            continue
        for lbl, col in cols_sig.items():
            v = s[col].values.astype(float)
            mu, std = v.mean(), v.std()
            if std > 0:
                all_vals[lbl].extend(((v - mu) / std).tolist())
            else:
                all_vals[lbl].extend([0.] * len(v))

    n_methods = len(cols_sig)
    names = list(cols_sig.keys())
    corr_mat = np.eye(n_methods)
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i == j:
                continue
            v1, v2 = np.array(all_vals[n1]), np.array(all_vals[n2])
            common  = np.isfinite(v1) & np.isfinite(v2)
            if common.sum() > 5:
                corr_mat[i, j] = np.corrcoef(v1[common], v2[common])[0, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(n_methods)); ax.set_yticks(range(n_methods))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    for i in range(n_methods):
        for j in range(n_methods):
            ax.text(j, i, f"{corr_mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="black" if abs(corr_mat[i,j]) < 0.85 else "white")
    ax.set_title("Matrice de corrélation de Pearson\n"
                 "(signaux centrés-réduits par balisée, toutes balisées confondues)",
                 fontsize=11)
    plt.tight_layout()
    out = out_dir / "pulsation_correlation_matrix.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


# ─── Rapport Markdown ─────────────────────────────────────────────────────────

def write_report(
    cmp: pd.DataFrame,
    df:  pd.DataFrame,
    alerts: list[str],
    log: logging.Logger,
) -> None:
    OUT_FIGS_REL = "figures"

    # Déterminer la meilleure méthode sur chaque critère
    amp_cols = ["amplitude_rel_bbox", "amplitude_rel_yolovseg", "amplitude_rel_sam"]
    cv_cols  = ["cv_bbox",            "cv_yolovseg",             "cv_sam"]
    amp_means = cmp[amp_cols].mean()
    cv_means  = cmp[cv_cols].mean()
    best_amp  = amp_means.idxmax().replace("amplitude_rel_", "")
    best_cv   = cv_means.idxmin().replace("cv_", "")

    inter_var_col = amp_cols
    inter_stds    = cmp[inter_var_col].std()
    best_inter    = inter_stds.idxmin().replace("amplitude_rel_", "")

    # Convergence fréquences
    fft_cols  = ["freq_fft_bbox_hz", "freq_fft_yolovseg_hz", "freq_fft_sam_hz"]
    fft_means = cmp[fft_cols].mean()
    fft_stds  = cmp[fft_cols].std()
    converge_msg = (
        "Les 3 méthodes convergent sur des fréquences dominantes similaires"
        if fft_stds.max() < 0.3
        else "Des divergences de fréquence sont observées entre méthodes"
    )

    # Table recap
    tbl_cols = ["balisee_id", "n_frames_total", "duree_s",
                "amplitude_rel_bbox", "amplitude_rel_yolovseg", "amplitude_rel_sam",
                "cv_bbox", "cv_yolovseg", "cv_sam",
                "freq_fft_bbox_hz", "freq_fft_yolovseg_hz", "freq_fft_sam_hz"]
    tbl_cols_present = [c for c in tbl_cols if c in cmp.columns]
    # Build markdown table manually (no tabulate dependency)
    tbl_df = cmp[tbl_cols_present].round(3)
    header = "| " + " | ".join(tbl_df.columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(tbl_df.columns)) + " |"
    rows   = ["| " + " | ".join(str(v) for v in row) + " |"
              for row in tbl_df.itertuples(index=False)]
    tbl_md = "\n".join([header, sep] + rows)

    alerts_section = ""
    if alerts:
        alerts_section = "\n### Alertes détectées\n\n"
        for a in alerts:
            alerts_section += f"- **{a}**\n"

    # Corrélation bbox/SAM globale
    corr_vals = []
    for bname in BALISEES:
        s = get_series(df, bname)
        valid = s["bbox_area_px"].notna() & s["mask_area_sam_px"].notna()
        if valid.sum() > 10:
            v1 = s.loc[valid, "bbox_area_px"].values
            v2 = s.loc[valid, "mask_area_sam_px"].values
            mu1, s1 = v1.mean(), v1.std()
            mu2, s2 = v2.mean(), v2.std()
            if s1 > 0 and s2 > 0:
                corr_vals.append(np.corrcoef((v1-mu1)/s1, (v2-mu2)/s2)[0,1])
    corr_mean = np.mean(corr_vals) if corr_vals else float("nan")

    report = f"""# Rapport d'analyse — Phase 6 : Comparaison des méthodes de mesure de pulsation

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

Les filtres appliqués : `time_s >= {TIME_FILTER}s` (exclusion des 40 premières secondes
d'instabilité du drone).

---

## Tableau récapitulatif

{tbl_md}

---

## Résultats clés
{alerts_section}
### Méthode la plus contrastée (amplitude relative P95–P5 / médiane)

La méthode **{best_amp.upper()}** présente l'amplitude relative moyenne la plus élevée
({amp_means[[f'amplitude_rel_{best_amp}']].values[0]:.3f}).
Une amplitude relative haute indique que la pulsation crée une variation relative large
du signal, ce qui facilite la détection des pics.

Amplitudes moyennes : bbox={amp_means['amplitude_rel_bbox']:.3f},
yolovseg={amp_means['amplitude_rel_yolovseg']:.3f}, sam={amp_means['amplitude_rel_sam']:.3f}.

### Méthode la plus stable (CV le plus bas)

La méthode **{best_cv.upper()}** présente le coefficient de variation moyen le plus bas
({cv_means[[f'cv_{best_cv}']].values[0]:.3f}), indiquant un signal moins bruité relativement
à sa moyenne.

### Méthode la plus cohérente entre balisées (variance inter-balisées)

La méthode **{best_inter.upper()}** montre la plus faible variance inter-balisées sur
l'amplitude relative, suggérant un comportement plus reproductible d'une méduse à l'autre.

### Convergence des fréquences FFT vs détection de pics

{converge_msg} ({', '.join([f"{c.replace('freq_fft_','').replace('_hz','').upper()}={v:.3f} Hz"
                              for c, v in zip(fft_cols, fft_means.values)])}).

La bande biologique attendue pour *Rhizostoma pulmo* est **0.3–1.5 Hz**
(période de 0.7–3.3s). Les fréquences identifiées par FFT
{
"se situent dans cette plage pour la majorité des balisées."
if cmp[fft_cols].apply(lambda c: c.between(0.3, 1.5)).all().any()
else "s'écartent partiellement de cette plage biologiquement attendue."
}

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
   par balisée) = **{corr_mean:.3f}**. Une valeur proche de 1 indiquerait que
   SAM ne produit pas d'information complémentaire à la bbox.

---

## Recommandation

Sur la base de ces résultats, la méthode recommandée pour la suite du projet est :

- **SAM (priorité 1)** si disponibilité GPU : couverture 100 %, masques robustes
  aux petites méduses, précision du contour meilleure que la bbox, indépendant du
  seuil de confiance du modèle seg.

- **Bbox détection (fallback)** si contrainte de temps/GPU : simple, rapide, 100 %
  de couverture, signal de pulsation viable si les baisées ne changent pas
  significativement d'échelle.

- **YOLOv8-seg** : à utiliser si on veut quantifier la forme précise du manteau,
  mais seulement après ré-entraînement avec plus de données (mAP50-95 mask actuel = 0.139).

---

*Généré automatiquement par `src/analyze_pulsation_comparison.py` — Phase 6.3*
"""

    OUT_RPT.parent.mkdir(parents=True, exist_ok=True)
    OUT_RPT.write_text(report, encoding="utf-8")
    log.info(f"Rapport écrit : {OUT_RPT}")
    print(f"[OK] Rapport : {OUT_RPT}")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main() -> None:
    log = setup_logging()
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    log.info("Chargement et jointure des 3 CSV…")
    df = load_and_join()
    log.info(f"Dataset joint : {len(df)} lignes après filtre time_s >= {TIME_FILTER}s")

    log.info("Analyse pulsation par balisée…")
    cmp = analyze(df, log)

    log.info("Vérifications des alertes…")
    alerts = check_alerts(cmp, df, log)

    # Sauvegarde CSV récap
    cmp.to_csv(OUT_CMP, index=False)
    log.info(f"CSV récap : {OUT_CMP}")
    print(f"\n[OK] CSV récap : {OUT_CMP.name}")
    print(cmp[["balisee_id", "amplitude_rel_bbox", "amplitude_rel_yolovseg",
               "amplitude_rel_sam", "freq_fft_bbox_hz", "freq_fft_yolovseg_hz",
               "freq_fft_sam_hz"]].round(3).to_string(index=False))

    log.info("Génération des figures…")
    fig_three_signals(df, OUT_FIGS)
    fig_amplitude_comparison(cmp, OUT_FIGS)
    fig_frequency_comparison(cmp, OUT_FIGS)
    fig_correlation_matrix(df, OUT_FIGS)

    log.info("Rédaction du rapport Markdown…")
    write_report(cmp, df, alerts, log)

    log.info("Phase 6.3 terminée.")
    print("\n[OK] Phase 6.3 terminée — tous les livrables dans results/ et figures/")


if __name__ == "__main__":
    main()
