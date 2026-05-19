"""
Phase 6.3 — Vérification fréquentielle.

Pour chaque (balisée, méthode) :
  - spectre FFT complet [0, Nyquist], top-3 pics, ratio P2/P1
  - figure 5×3 lin-lin + inset log-y

Pour B3 (signal de référence) :
  - comptage oscillations par passages à zéro (signal lissé centré)
  - comparaison avec FFT × durée

Ajoute une section "Vérification fréquentielle" à
  results/pulsation_analysis_report.md

Log dans logs/phase6_analysis.log (append).
"""

import logging
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

# ── Chemins ────────────────────────────────────────────────────────────────────
CSV_DET  = REPO / "results" / "trajectories_detection_DJI0013.csv"
CSV_SEG  = REPO / "results" / "trajectories_seg_yolovseg_DJI0013.csv"
CSV_SAM  = REPO / "results" / "trajectories_seg_sam_DJI0013.csv"
OUT_FIGS = REPO / "figures"
OUT_RPT  = REPO / "results" / "pulsation_analysis_report.md"
LOG_PATH = REPO / "logs" / "phase6_analysis.log"

FPS_SOURCE  = 29.97
VID_STRIDE  = 6
EFF_FPS     = FPS_SOURCE / VID_STRIDE    # ~4.995 Hz
NYQUIST     = EFF_FPS / 2               # ~2.497 Hz
TIME_FILTER = 40.0
BIO_LO, BIO_HI = 0.3, 1.5

BALISEES: dict[str, list[int]] = {
    "Balisee_1": [767, 936, 946, 958, 1165, 1217],
    "Balisee_2": [771, 1220, 1371],
    "Balisee_3": [764],
    "Balisee_4": [787, 863, 942, 981, 1038, 1193, 1257, 1344],
    "Balisee_5": [847, 945, 1178, 1237, 1288],
}
BALISEE_REF = "Balisee_3"

METHODS = {
    "bbox":     "bbox_area_px",
    "yolovseg": "mask_area_yolovseg_px",
    "sam":      "mask_area_sam_px",
}
METHOD_LABELS = {"bbox": "BBox", "yolovseg": "YOLOv8-seg", "sam": "SAM"}
METHOD_COLORS = {"bbox": "#e74c3c", "yolovseg": "#2ecc71", "sam": "#9b59b6"}


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        ],
    )
    return logging.getLogger("phase6_fft_verify")


# ── Chargement ─────────────────────────────────────────────────────────────────

def load_and_join() -> pd.DataFrame:
    det = pd.read_csv(CSV_DET)
    seg = pd.read_csv(CSV_SEG)[["frame_idx", "jelly_id",
                                 "mask_area_yolovseg_px"]]
    sam = pd.read_csv(CSV_SAM)[["frame_idx", "jelly_id",
                                 "mask_area_sam_px"]]
    df = (det
          .merge(seg, on=["frame_idx", "jelly_id"], how="left")
          .merge(sam, on=["frame_idx", "jelly_id"], how="left"))
    return df[df["time_s"] >= TIME_FILTER].copy()


def get_series(df: pd.DataFrame, bname: str) -> pd.DataFrame:
    ids = BALISEES[bname]
    sub = df[df["jelly_id"].isin(ids)].copy()
    sub = sub.sort_values("time_s").drop_duplicates("time_s", keep="first")
    return sub.reset_index(drop=True)


# ── Signal préparation ─────────────────────────────────────────────────────────

def prepare_signal(s: pd.DataFrame, col: str):
    """Retourne (t, v_smooth, dt) après filtrage NaN et lissage médiane k=3."""
    v_raw = s[col].values.astype(float)
    t_all = s["time_s"].values.astype(float)
    valid = np.isfinite(v_raw)
    if valid.sum() < 10:
        return None, None, None
    v = v_raw[valid]
    t = t_all[valid]
    v_smooth = median_filter(v, size=3, mode="nearest")
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / EFF_FPS
    if dt <= 0:
        dt = 1.0 / EFF_FPS
    return t, v_smooth, dt


# ── Analyse FFT ────────────────────────────────────────────────────────────────

def fft_analysis(v_smooth: np.ndarray, dt: float) -> dict:
    """
    FFT complète sur signal centré.
    Retourne freqs, power, top3 pics (freq, puissance), ratio P2/P1.
    """
    v_c = v_smooth - v_smooth.mean()
    freqs = np.fft.rfftfreq(len(v_c), d=dt)
    power = np.abs(np.fft.rfft(v_c)) ** 2

    # Restreindre à [0, Nyquist] — rfftfreq ne dépasse pas Nyquist par construction
    # Exclure DC (freq=0) et très basses fréquences (<0.05 Hz : dérive lente)
    valid_mask = freqs >= 0.05
    freqs_v = freqs[valid_mask]
    power_v = power[valid_mask]

    # Détection de pics dans le spectre (distance min = 3 bins)
    peaks_idx, _ = sp_signal.find_peaks(power_v, distance=3)

    if len(peaks_idx) == 0:
        # fallback : argmax simple
        peaks_idx = np.array([np.argmax(power_v)])

    # Trier par puissance décroissante
    order = np.argsort(power_v[peaks_idx])[::-1]
    top_idx = peaks_idx[order[:3]]

    top_freqs  = freqs_v[top_idx]
    top_powers = power_v[top_idx]

    # Ratio P2/P1 et P3/P1
    ratio_p2p1 = float(top_powers[1] / top_powers[0]) if len(top_powers) > 1 else np.nan
    ratio_p3p1 = float(top_powers[2] / top_powers[0]) if len(top_powers) > 2 else np.nan

    # Fréquence dominante dans bande biologique complète [BIO_LO, NYQUIST]
    bio_mask = (freqs_v >= BIO_LO) & (freqs_v <= NYQUIST)
    if bio_mask.any():
        freq_bio = float(freqs_v[bio_mask][np.argmax(power_v[bio_mask])])
        power_bio = float(power_v[bio_mask].max())
    else:
        freq_bio  = float(top_freqs[0]) if len(top_freqs) > 0 else np.nan
        power_bio = float(top_powers[0]) if len(top_powers) > 0 else np.nan

    # Fréquence dominante hors-DC (toute la bande)
    freq_dom = float(freqs_v[np.argmax(power_v)])

    return {
        "freqs":       freqs_v,
        "power":       power_v,
        "freqs_full":  freqs,
        "power_full":  power,
        "top_freqs":   top_freqs,
        "top_powers":  top_powers,
        "ratio_p2p1":  ratio_p2p1,
        "ratio_p3p1":  ratio_p3p1,
        "freq_dom":    freq_dom,
        "freq_bio":    freq_bio,
        "power_bio":   power_bio,
    }


# ── Comptage oscillations par passages à zéro (B3) ────────────────────────────

def count_oscillations_zero_crossing(v_smooth: np.ndarray) -> int:
    """
    Compte le nombre d'oscillations complètes par passages à zéro du signal centré.
    1 oscillation = 2 passages montants ou 2 croisements (+ → -) successifs.
    On compte les passages montants (- → +) du signal centré.
    """
    v_c = v_smooth - v_smooth.mean()
    # Passages de - à + (front montant)
    signs = np.sign(v_c)
    crossings = np.where((signs[:-1] <= 0) & (signs[1:] > 0))[0]
    return len(crossings)


def count_peaks_loose(v_smooth: np.ndarray, dt: float) -> int:
    """Détection de pics avec contraintes relâchées (distance min = 0.3s)."""
    min_dist = max(int(EFF_FPS * 0.3), 1)
    med = np.median(v_smooth)
    prom_min = max(0.05 * v_smooth.std(), 0.5)
    peaks, _ = sp_signal.find_peaks(
        v_smooth,
        height=med,
        distance=min_dist,
        prominence=prom_min,
    )
    return len(peaks)


# ── Figure 5×3 spectres FFT ────────────────────────────────────────────────────

def fig_fft_spectra(results: dict, out_dir: Path) -> Path:
    """
    Grille 5 (balisées) × 3 (méthodes).
    Chaque cellule : spectre lin-lin principal + inset log-y.
    Top-3 pics annotés.
    """
    bnames  = list(BALISEES.keys())
    methods = list(METHODS.keys())

    fig, axes = plt.subplots(
        5, 3, figsize=(18, 22),
        constrained_layout=True,
    )
    fig.suptitle(
        "Spectres FFT de puissance — 5 balisées × 3 méthodes\n"
        f"Bande biologique *Rhizostoma pulmo* [{BIO_LO}–{BIO_HI} Hz] grisée",
        fontsize=13, fontweight="bold",
    )

    for row, bname in enumerate(bnames):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            key = (bname, method)

            if key not in results or results[key] is None:
                ax.text(0.5, 0.5, "Données insuffisantes",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{bname} / {METHOD_LABELS[method]}", fontsize=8)
                continue

            r = results[key]
            freqs  = r["freqs"]
            power  = r["power"]
            color  = METHOD_COLORS[method]

            # ── Lin-lin principal ──────────────────────────────────────────
            ax.fill_between(
                [BIO_LO, min(BIO_HI, NYQUIST)],
                0, power.max() * 1.05,
                alpha=0.10, color="#f39c12", label="Bande bio",
            )
            ax.plot(freqs, power, color=color, lw=0.9, alpha=0.85)

            # Annotate top-3 peaks
            top_f = r["top_freqs"]
            top_p = r["top_powers"]
            markers = ["▲", "▼", "◆"]
            for rank, (tf, tp) in enumerate(zip(top_f, top_p)):
                mk_color = "#c0392b" if rank == 0 else "#7f8c8d"
                ax.plot(tf, tp, "^" if rank == 0 else "v",
                        color=mk_color, ms=5 if rank == 0 else 3.5, zorder=5)
                ax.annotate(
                    f"P{rank+1}={tf:.3f}Hz",
                    xy=(tf, tp),
                    xytext=(4, 4 - rank * 10),
                    textcoords="offset points",
                    fontsize=6,
                    color=mk_color,
                )

            # Ratio P2/P1 text
            r2p1 = r["ratio_p2p1"]
            if np.isfinite(r2p1):
                flag = " ⚠" if r2p1 > 0.5 else ""
                ax.text(
                    0.97, 0.95, f"P2/P1={r2p1:.2f}{flag}",
                    ha="right", va="top", transform=ax.transAxes,
                    fontsize=6.5, color="#8e44ad" if r2p1 > 0.5 else "#555",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
                )

            ax.set_xlim(0, NYQUIST)
            ax.set_ylim(bottom=0)
            ax.axvline(BIO_LO, color="#f39c12", lw=0.6, ls="--", alpha=0.5)
            ax.axvline(min(BIO_HI, NYQUIST), color="#f39c12", lw=0.6, ls="--", alpha=0.5)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("Fréquence (Hz)", fontsize=6.5)
            ax.set_ylabel("Puissance", fontsize=6.5)

            ref_tag = " ★ref" if bname == BALISEE_REF else ""
            ax.set_title(
                f"{bname}{ref_tag} / {METHOD_LABELS[method]}",
                fontsize=8, pad=3,
            )

            # ── Inset log-y ────────────────────────────────────────────────
            ax_in = ax.inset_axes([0.60, 0.40, 0.38, 0.55])
            ax_in.semilogy(freqs, np.maximum(power, 1e-10),
                           color=color, lw=0.7, alpha=0.8)
            for tf, tp in zip(top_f[:1], top_p[:1]):
                ax_in.axvline(tf, color="#c0392b", lw=0.8, ls="--", alpha=0.7)
            ax_in.axvspan(BIO_LO, min(BIO_HI, NYQUIST), alpha=0.08, color="#f39c12")
            ax_in.set_xlim(0, NYQUIST)
            ax_in.tick_params(labelsize=4.5)
            ax_in.set_title("log-y", fontsize=4.5, pad=1)

    out_path = out_dir / "pulsation_fft_spectra.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_path}")
    return out_path


# ── Vérification B3 détaillée ──────────────────────────────────────────────────

def verify_b3(df: pd.DataFrame, log: logging.Logger) -> dict:
    """
    Sur B3 (signal le plus propre, 100 % YOLOseg coverage) :
    - Comptage passages à zéro du signal bbox centré-lissé
    - Comptage pics avec critères relâchés
    - Comparaison avec FFT × durée
    """
    s = get_series(df, BALISEE_REF)
    t, v_smooth, dt = prepare_signal(s, "bbox_area_px")
    if t is None:
        return {}

    duree_s = float(t[-1] - t[0])
    r = fft_analysis(v_smooth, dt)

    n_zero_cross = count_oscillations_zero_crossing(v_smooth)
    n_peaks_loose = count_peaks_loose(v_smooth, dt)
    freq_fft_dom  = r["freq_dom"]
    freq_fft_bio  = r["freq_bio"]
    n_expected_fft_dom = round(freq_fft_dom * duree_s)
    n_expected_fft_bio = round(freq_fft_bio * duree_s)

    log.info(
        f"B3 bbox | durée={duree_s:.1f}s | "
        f"ZC={n_zero_cross} osc | pics relâchés={n_peaks_loose} | "
        f"FFT dom={freq_fft_dom:.4f} Hz → {n_expected_fft_dom} osc attendues | "
        f"FFT bio={freq_fft_bio:.4f} Hz → {n_expected_fft_bio} osc attendues"
    )

    # Discordance : si écart > 50 % entre comptage réel et FFT
    ratio_zc_fft = n_zero_cross / n_expected_fft_dom if n_expected_fft_dom > 0 else np.nan
    discordance = abs(ratio_zc_fft - 1.0) > 0.5 if np.isfinite(ratio_zc_fft) else True

    return {
        "duree_s":           duree_s,
        "n_zero_cross":      n_zero_cross,
        "n_peaks_loose":     n_peaks_loose,
        "freq_fft_dom":      freq_fft_dom,
        "freq_fft_bio":      freq_fft_bio,
        "n_expected_fft_dom": n_expected_fft_dom,
        "n_expected_fft_bio": n_expected_fft_bio,
        "ratio_zc_fft":      ratio_zc_fft,
        "discordance":       discordance,
        "top3_freqs":        r["top_freqs"],
        "top3_powers":       r["top_powers"],
        "ratio_p2p1":        r["ratio_p2p1"],
    }


# ── Conclusion harmonique ──────────────────────────────────────────────────────

def interpret_frequency(results: dict, b3: dict, log: logging.Logger) -> str:
    """
    Détermine si 0.23 Hz est la fondamentale ou une sous-harmonique.

    Critères :
    1. Si ratio_zc_fft ≈ 1 (± 50 %) → FFT cohérente avec le signal brut
       → 0.23 Hz = fondamentale (méduses au repos)
    2. Si ratio_zc_fft >> 1 → il y a plus d'oscillations que prévu par FFT
       → 0.23 Hz est une sous-harmonique, fondamentale plus haute
    3. Si ratio_p2p1 > 0.5 sur plusieurs balisées → signal multi-composantes
       → possibilité de sous-échantillonnage de la fondamentale
    """
    multi_comp_count = sum(
        1 for (bn, m), r in results.items()
        if r is not None and np.isfinite(r["ratio_p2p1"]) and r["ratio_p2p1"] > 0.5
    )
    total_pairs = sum(1 for r in results.values() if r is not None)

    ratio_zc = b3.get("ratio_zc_fft", np.nan)
    r_p2p1_b3_bbox = results.get((BALISEE_REF, "bbox"), {})
    r2p1 = r_p2p1_b3_bbox.get("ratio_p2p1", np.nan) if r_p2p1_b3_bbox else np.nan

    log.info(
        f"Interprétation : {multi_comp_count}/{total_pairs} paires avec ratio_p2p1>0.5 | "
        f"B3 ZC/FFT={ratio_zc:.2f} | B3 bbox ratio_p2p1={r2p1:.2f}"
    )

    if np.isfinite(ratio_zc) and 0.5 <= ratio_zc <= 1.5:
        # Cohérent → fondamentale
        conclusion = "fondamentale"
        detail = (
            f"Le comptage par passages à zéro ({b3['n_zero_cross']} oscillations sur "
            f"{b3['duree_s']:.0f}s) est cohérent avec la fréquence FFT dominante "
            f"({b3['freq_fft_dom']:.3f} Hz × {b3['duree_s']:.0f}s ≈ "
            f"{b3['n_expected_fft_dom']} oscillations attendues, "
            f"ratio={ratio_zc:.2f}). "
            "La fréquence ~0.23 Hz correspond vraisemblablement à la **fondamentale** : "
            "les méduses pulsent lentement (~14 fois par minute), "
            "comportement typique d'individus au repos ou sous faible effort locomoteur."
        )
    elif np.isfinite(ratio_zc) and ratio_zc > 1.5:
        # Plus d'oscillations que FFT ne l'indique → sous-harmonique
        conclusion = "sous-harmonique"
        detail = (
            f"Le comptage par passages à zéro ({b3['n_zero_cross']} oscillations) est "
            f"{ratio_zc:.1f}× supérieur aux oscillations attendues par la FFT "
            f"({b3['n_expected_fft_dom']}). "
            "La fréquence FFT dominante (~0.23 Hz) est probablement une **sous-harmonique** "
            "ou reflète une composante basse fréquence (dérive lente, mouvement du drone). "
            "La fondamentale réelle serait plus proche de "
            f"~{b3['freq_fft_bio']:.3f} Hz (pic dans la bande biologique)."
        )
    else:
        conclusion = "indéterminé"
        detail = (
            "Les données sont insuffisantes pour trancher de manière définitive. "
            "Le ratio passages-à-zéro / FFT est hors plage ou non calculable."
        )

    # Complément sur signal multi-composantes
    if multi_comp_count >= total_pairs // 2:
        detail += (
            f"\n\n**Signal multi-composantes** : {multi_comp_count}/{total_pairs} paires "
            "présentent un ratio P2/P1 > 0.5, indiquant que le spectre a plusieurs "
            "composantes significatives. Cela peut résulter du mélange du signal de "
            "pulsation avec des oscillations lentes du drone (typiquement 0.05–0.2 Hz)."
        )

    return conclusion, detail


# ── Génération de la section rapport ──────────────────────────────────────────

def build_report_section(
    results: dict,
    b3: dict,
    conclusion: str,
    detail: str,
) -> str:
    """Construit le texte markdown de la section 'Vérification fréquentielle'."""

    bnames  = list(BALISEES.keys())
    methods = list(METHODS.keys())

    # Table résumé top-3 par (balisée, méthode)
    rows = []
    for bname in bnames:
        for method in methods:
            key = (bname, method)
            ref_tag = " ★" if bname == BALISEE_REF else ""
            if key not in results or results[key] is None:
                rows.append(f"| {bname}{ref_tag} | {METHOD_LABELS[method]} | — | — | — | — | — |")
                continue
            r = results[key]
            top_f = r["top_freqs"]
            top_p = r["top_powers"]
            in_bio = [BIO_LO <= f <= BIO_HI for f in top_f]

            def fmt_peak(i):
                if i >= len(top_f):
                    return "—"
                tag = " ✓" if in_bio[i] else ""
                return f"{top_f[i]:.3f} Hz{tag}"

            r2p1 = r["ratio_p2p1"]
            r2p1_str = f"{r2p1:.2f} {'⚠' if r2p1 > 0.5 else ''}" if np.isfinite(r2p1) else "—"
            rows.append(
                f"| {bname}{ref_tag} | {METHOD_LABELS[method]} "
                f"| {fmt_peak(0)} | {fmt_peak(1)} | {fmt_peak(2)} | {r2p1_str} |"
            )

    table = "\n".join([
        "| Balisée | Méthode | Pic 1 (dominant) | Pic 2 | Pic 3 | P2/P1 |",
        "|---------|---------|-----------------|-------|-------|-------|",
    ] + rows)

    # Vérification B3
    b3_section = ""
    if b3:
        b3_section = f"""
### Vérification sur B3 (signal de référence — 100 % couverture YOLOseg)

| Indicateur | Valeur |
|-----------|--------|
| Durée analysée | {b3['duree_s']:.1f} s |
| Passages à zéro (oscillations comptées) | **{b3['n_zero_cross']}** |
| Pics détectés (critères relâchés) | {b3['n_peaks_loose']} |
| Fréquence FFT dominante | {b3['freq_fft_dom']:.4f} Hz |
| Oscillations attendues (FFT dom × durée) | {b3['n_expected_fft_dom']} |
| Ratio passages-à-zéro / attendu FFT | **{b3['ratio_zc_fft']:.2f}** |
| Fréquence FFT bande bio ([{BIO_LO}–{BIO_HI} Hz]) | {b3['freq_fft_bio']:.4f} Hz |
| Top-3 fréquences spectrales | {', '.join(f'{f:.3f} Hz' for f in b3['top3_freqs'])} |
| P2/P1 spectre B3-bbox | {b3['ratio_p2p1']:.2f} |

Méthode de comptage : passages montants (−→+) du signal `bbox_area_px` centré-réduit
et lissé (médiane mobile k=3), sur la totalité de la fenêtre d'analyse (t ≥ 40 s).
"""

    conclusion_label = {
        "fondamentale":  "**0.23 Hz = fondamentale → comportement au repos**",
        "sous-harmonique": "**0.23 Hz = sous-harmonique → fondamentale réelle plus haute**",
        "indéterminé": "**Indéterminé — données insuffisantes**",
    }.get(conclusion, conclusion)

    section = f"""

---

## Vérification fréquentielle

> Section ajoutée après l'analyse 6.3 pour vérifier si les fréquences FFT
> (~0.23–0.30 Hz, sous la bande biologique 0.3–1.5 Hz) correspondent à la
> fondamentale de pulsation ou à une composante de basse fréquence parasite.

### Paramètres d'analyse

- Fréquence d'échantillonnage effective : {EFF_FPS:.3f} Hz (vid_stride={int(VID_STRIDE)})
- Fréquence de Nyquist : {NYQUIST:.3f} Hz
- Bande biologique *Rhizostoma pulmo* : [{BIO_LO}–{BIO_HI}] Hz
- Spectres calculés sur signal centré (DC retiré) après médiane mobile k=3
- Top-3 pics : détection scipy.signal.find_peaks sur le spectre de puissance, distance ≥ 3 bins
- ✓ = fréquence dans la bande biologique | ⚠ = ratio P2/P1 > 0.5

### Tableau des 3 principaux pics spectraux (toutes balisées × méthodes)

{table}

*(★ = balisée de référence B3 — 100 % couverture YOLOseg)*
{b3_section}
### Conclusion : {conclusion_label}

{detail}

### Figure associée

`figures/pulsation_fft_spectra.png` — Grille 5×3, spectres lin-lin avec insets log-y,
top-3 pics annotés, bande biologique grisée.

---
"""
    return section


# ── Ajout au rapport ───────────────────────────────────────────────────────────

def append_to_report(section: str, log: logging.Logger):
    if not OUT_RPT.exists():
        log.warning(f"Rapport introuvable : {OUT_RPT} — section non ajoutée.")
        return
    content = OUT_RPT.read_text(encoding="utf-8")
    # Si section déjà présente, remplacer
    marker = "\n---\n\n## Vérification fréquentielle"
    if marker in content:
        before = content.split(marker)[0]
        # Trouver la fin de cette section (prochain "---\n" de niveau section)
        rest_after = content[content.index(marker):]
        # Chercher la prochaine section H2 après la notre, ou fin de fichier
        next_h2 = rest_after.find("\n## ", 5)
        next_hr_after_section = rest_after.find("\n---\n", len(marker))
        if next_h2 > 0:
            after = rest_after[next_h2:]
        elif next_hr_after_section > 0 and next_hr_after_section > len(section):
            after = rest_after[next_hr_after_section:]
        else:
            # garde juste la ligne signature finale
            after = "\n*Généré automatiquement" + rest_after.split("*Généré automatiquement")[-1]
        new_content = before + section + after
    else:
        # Insérer avant la ligne de signature finale
        sig = "\n*Généré automatiquement"
        if sig in content:
            idx = content.rfind(sig)
            new_content = content[:idx] + section + content[idx:]
        else:
            new_content = content + section
    OUT_RPT.write_text(new_content, encoding="utf-8")
    log.info(f"Section 'Vérification fréquentielle' ajoutée à {OUT_RPT}")
    print(f"[OK] Rapport mis à jour : {OUT_RPT}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log = setup_logging()
    log.info("=== Vérification fréquentielle (Phase 6.3 add-on) ===")

    log.info("Chargement des données…")
    df = load_and_join()
    log.info(f"Dataset : {len(df)} lignes (t ≥ {TIME_FILTER}s)")

    # ── Calcul FFT pour toutes les paires (balisée, méthode) ──────────────────
    results = {}
    log.info("Calcul des spectres FFT…")
    for bname in BALISEES:
        s = get_series(df, bname)
        for method, col in METHODS.items():
            t, v_smooth, dt = prepare_signal(s, col)
            if t is None:
                log.warning(f"  {bname}/{method} : signal insuffisant")
                results[(bname, method)] = None
                continue
            r = fft_analysis(v_smooth, dt)
            results[(bname, method)] = r
            log.info(
                f"  {bname}/{method} : "
                f"dom={r['freq_dom']:.4f} Hz | "
                f"bio={r['freq_bio']:.4f} Hz | "
                f"top3=[{', '.join(f'{f:.3f}' for f in r['top_freqs'])}] Hz | "
                f"P2/P1={r['ratio_p2p1']:.2f}"
            )

    # ── Vérification B3 ───────────────────────────────────────────────────────
    log.info(f"Vérification détaillée {BALISEE_REF}…")
    b3 = verify_b3(df, log)

    # ── Interprétation ────────────────────────────────────────────────────────
    conclusion, detail = interpret_frequency(results, b3, log)
    log.info(f"Conclusion : {conclusion}")

    # ── Figure ────────────────────────────────────────────────────────────────
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    log.info("Génération figure pulsation_fft_spectra.png…")
    fig_fft_spectra(results, OUT_FIGS)

    # ── Rapport ───────────────────────────────────────────────────────────────
    section = build_report_section(results, b3, conclusion, detail)
    append_to_report(section, log)

    log.info("Vérification fréquentielle terminée.")
    print("\n[OK] Vérification fréquentielle terminée — figures/ et rapport mis à jour.")


if __name__ == "__main__":
    main()
