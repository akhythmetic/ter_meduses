"""
Phase 6 — Test haute fréquence SAM sur B3.

Lance la détection + SAM sur les frames 4193–4793 de DJI_0013 (20s)
avec vid_stride=1 (29.97 Hz natif). YOLOv8-seg et ByteTrack sont
volontairement omis : on mesure uniquement mask_area_sam par frame.

Identifie B3 via un filtre spatial autour de sa position connue dans
ce segment (~2733, ~1218 px). Produit :
  results/trajectories_seg_sam_DJI0013_hifreq.csv
  figures/pulsation_hifreq_B3.png

Usage :
    python src/inference_sam_hifreq.py --video "TER analyse videos meduses/2025_06_30/DJI_0013.MP4"
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent

DEFAULT_VIDEO  = REPO / "TER analyse videos meduses" / "2025_06_30" / "DJI_0013.MP4"
DEFAULT_DETECT = REPO / "runs" / "detect" / "yolov8n_1280_e150" / "weights" / "best.pt"
DEFAULT_SAM    = REPO / "sam2.1_b.pt"

OUT_CSV = REPO / "results" / "trajectories_seg_sam_DJI0013_hifreq.csv"
OUT_FIG = REPO / "figures" / "pulsation_hifreq_B3.png"

FRAME_START = 4193
FRAME_END   = 4793
FPS_SOURCE  = 29.97
IMGSZ       = 1280
CONF        = 0.25

# Région spatiale de B3 dans ce segment (d'après les données stride=6)
B3_CX = 2733.0
B3_CY = 1218.0
B3_TOL_X = 150.0   # ±px autour du centre X connu
B3_TOL_Y = 120.0   # ±px autour du centre Y connu

COLS_CSV = ["frame_idx", "time_s", "x_center", "y_center",
            "width", "height", "bbox_area_px",
            "mask_area_sam_px", "mask_perimeter_sam_px"]


def mask_metrics(mask_u8: np.ndarray) -> tuple[int, float]:
    area = int(mask_u8.sum())
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    perim = sum(cv2.arcLength(c, closed=True) for c in contours)
    return area, float(perim)


def resize_mask(mask_arr: np.ndarray, h: int, w: int) -> np.ndarray:
    m = mask_arr.astype(np.uint8)
    if m.shape[0] == h and m.shape[1] == w:
        return m
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def run_inference(video_path: Path, detect_path: Path, sam_path: Path) -> None:
    from ultralytics import SAM, YOLO

    print(f"Vidéo  : {video_path}")
    print(f"Detect : {detect_path}")
    print(f"SAM    : {sam_path}")
    print(f"Frames : {FRAME_START}–{FRAME_END}  ({FRAME_END - FRAME_START + 1} frames, {(FRAME_END - FRAME_START)/FPS_SOURCE:.1f}s)")
    print(f"B3 région : x∈[{B3_CX-B3_TOL_X:.0f},{B3_CX+B3_TOL_X:.0f}]  "
          f"y∈[{B3_CY-B3_TOL_Y:.0f},{B3_CY+B3_TOL_Y:.0f}]")

    for p in (video_path, detect_path, sam_path):
        if not p.exists():
            print(f"ERREUR : fichier introuvable : {p}", file=sys.stderr)
            sys.exit(1)

    print("Chargement modèles…")
    model_detect = YOLO(str(detect_path))
    sam_model    = SAM(str(sam_path))
    print("Modèles chargés.")

    cap = cv2.VideoCapture(str(video_path))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Résolution vidéo : {orig_w}×{orig_h}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_START)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    n_frames = FRAME_END - FRAME_START + 1

    rows_written = 0
    rows_b3      = 0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS_CSV)
        writer.writeheader()

        for fi in tqdm(range(n_frames), desc="HiFreq SAM", unit="fr"):
            frame_idx = FRAME_START + fi
            ret, frame = cap.read()
            if not ret:
                print(f"  Fin vidéo à frame {frame_idx}")
                break

            time_s = round(frame_idx / FPS_SOURCE, 5)

            # ── Détection YOLO (predict, pas track) ──────────────────────
            res_det = model_detect.predict(
                frame, imgsz=IMGSZ, conf=CONF, verbose=False
            )[0]

            if res_det.boxes is None or len(res_det.boxes) == 0:
                continue

            boxes_xyxy = res_det.boxes.xyxy.cpu().numpy()

            # ── Filtre spatial B3 ────────────────────────────────────────
            b3_bboxes = []
            b3_meta   = []
            for bi in range(len(res_det.boxes)):
                x1, y1, x2, y2 = boxes_xyxy[bi]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if (abs(cx - B3_CX) <= B3_TOL_X and
                        abs(cy - B3_CY) <= B3_TOL_Y):
                    b3_bboxes.append([x1, y1, x2, y2])
                    b3_meta.append((cx, cy, x2 - x1, y2 - y1))

            if not b3_bboxes:
                continue

            # ── SAM sur les candidats B3 ─────────────────────────────────
            try:
                sam_res = sam_model(frame, bboxes=b3_bboxes, verbose=False)
                if sam_res and sam_res[0].masks is not None:
                    masks_data = sam_res[0].masks.data.cpu().numpy()
                else:
                    masks_data = None
            except Exception as e:
                print(f"  SAM erreur frame {frame_idx}: {e}", file=sys.stderr)
                masks_data = None

            # Parmi les candidats, prend celui avec la plus grande aire SAM
            best_area  = -1
            best_row   = None

            for bi, (cx, cy, bw, bh) in enumerate(b3_meta):
                if masks_data is not None and bi < masks_data.shape[0]:
                    mask_u8 = resize_mask(masks_data[bi], orig_h, orig_w)
                    area, perim = mask_metrics(mask_u8)
                else:
                    area, perim = float("nan"), float("nan")

                if area > best_area:
                    best_area = area
                    best_row = dict(
                        frame_idx=frame_idx, time_s=time_s,
                        x_center=round(cx, 2), y_center=round(cy, 2),
                        width=round(bw, 2), height=round(bh, 2),
                        bbox_area_px=round(bw * bh, 2),
                        mask_area_sam_px=area,
                        mask_perimeter_sam_px=round(perim, 2),
                    )

            if best_row is not None:
                writer.writerow(best_row)
                rows_b3 += 1
            rows_written += 1

            if fi % 100 == 0:
                fh.flush()

    cap.release()
    print(f"\n[OK] CSV sauvegardé : {OUT_CSV}")
    print(f"     frames traitées : {rows_written}  |  B3 détections : {rows_b3}")


def analyze_and_plot() -> None:
    import pandas as pd

    df = pd.read_csv(OUT_CSV)
    df = df[df["mask_area_sam_px"].notna() & (df["mask_area_sam_px"] > 0)].copy()
    df = df.sort_values("time_s").reset_index(drop=True)

    if len(df) < 20:
        print(f"[WARN] Seulement {len(df)} points B3 — figure non générée")
        return

    t   = df["time_s"].values
    sig = df["mask_area_sam_px"].values.astype(float)

    # ── FFT ──────────────────────────────────────────────────────────────
    dt      = np.mean(np.diff(t))
    freqs   = np.fft.rfftfreq(len(sig), d=dt)
    fft_amp = np.abs(np.fft.rfft(sig - sig.mean()))

    # Fréquence dominante dans [0.1, 5] Hz
    mask_range = (freqs >= 0.1) & (freqs <= 5.0)
    if mask_range.any():
        dom_freq = float(freqs[mask_range][np.argmax(fft_amp[mask_range])])
    else:
        dom_freq = float(freqs[np.argmax(fft_amp[freqs > 0])])

    # Détection de pics dans le signal
    med = np.median(sig)
    std = np.std(sig)
    peaks, _ = sp_signal.find_peaks(
        sig,
        height=med + 0.3 * std,
        distance=max(int(FPS_SOURCE * 0.5), 1),
        prominence=max(0.15 * std, 1.0),
    )
    n_pics  = len(peaks)
    duree_s = float(t[-1] - t[0])
    freq_pics = n_pics / duree_s if duree_s > 0 else float("nan")

    print(f"\n=== Analyse haute fréquence B3 ===")
    print(f"  Points valides   : {len(df)}")
    print(f"  Durée analysée   : {duree_s:.1f}s")
    print(f"  fs effectif      : {1/dt:.2f} Hz")
    print(f"  Fréq. FFT dom.   : {dom_freq:.4f} Hz")
    print(f"  Fréq. par pics   : {freq_pics:.4f} Hz  ({n_pics} pics)")

    # ── Figure 2 sous-figures ────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle(
        f"B3 — signal SAM haute fréquence (29.97 Hz natif, vid_stride=1)\n"
        f"Segment frames {FRAME_START}–{FRAME_END}  ({t[0]:.1f}s – {t[-1]:.1f}s)",
        fontsize=13, fontweight="bold",
    )

    # (a) Signal temporel
    ax1.plot(t, sig, color="#e040fb", linewidth=0.7, alpha=0.85, label="mask_area_sam (brut)")
    if len(peaks):
        ax1.scatter(t[peaks], sig[peaks], color="#e040fb", s=30, zorder=5,
                    marker="v", edgecolors="white", linewidths=0.7,
                    label=f"Pics détectés (n={n_pics})")
    ax1.set_xlabel("Temps (s)", fontsize=10)
    ax1.set_ylabel("Aire masque SAM (px²)", fontsize=10)
    ax1.set_title(f"(a) Signal temporel — B3  |  fréq. pics = {freq_pics:.3f} Hz",
                  fontsize=11)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1000:.1f}k" if x >= 1000 else f"{x:.0f}")
    )
    ax1.legend(fontsize=9)
    ax1.yaxis.grid(True, alpha=0.3)

    # (b) Spectre FFT [0, 5 Hz]
    mask5 = freqs <= 5.0
    ax2.plot(freqs[mask5], fft_amp[mask5], color="#e040fb", linewidth=1.0, alpha=0.85)

    # Bande biologique grisée
    ax2.axvspan(0.3, 1.5, alpha=0.12, color="green",
                label="Bande biologique Rhizostoma (0.3–1.5 Hz)")

    # Annotation fréquence dominante
    ax2.axvline(dom_freq, color="#c62828", linewidth=1.5, linestyle="--",
                label=f"Fréq. dominante : {dom_freq:.3f} Hz")
    dom_amp = float(fft_amp[mask5][np.argmin(np.abs(freqs[mask5] - dom_freq))])
    ax2.annotate(
        f"{dom_freq:.3f} Hz",
        xy=(dom_freq, dom_amp),
        xytext=(dom_freq + 0.15, dom_amp * 0.9),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#c62828"),
        color="#c62828",
    )

    ax2.set_xlabel("Fréquence (Hz)", fontsize=10)
    ax2.set_ylabel("Amplitude FFT", fontsize=10)
    ax2.set_title(
        f"(b) Spectre FFT — B3  |  fréq. dominante = {dom_freq:.3f} Hz",
        fontsize=11,
    )
    ax2.legend(fontsize=9)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_xlim(0, 5)

    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Figure sauvegardée : {OUT_FIG}")

    return dom_freq, freq_pics, n_pics, duree_s


def append_report_section(dom_freq: float, freq_pics: float,
                           n_pics: int, duree_s: float) -> None:
    """Ajoute la section 'Vérification fréquentielle haute résolution' au rapport."""
    import pandas as pd

    rpt_path = REPO / "results" / "pulsation_analysis_report.md"
    rpt = rpt_path.read_text(encoding="utf-8")

    # Fréquence de référence stride=6 (B3 SAM)
    cmp_path = REPO / "results" / "pulsation_methods_comparison.csv"
    freq_stride6 = float("nan")
    if cmp_path.exists():
        cmp = pd.read_csv(cmp_path)
        b3_row = cmp[cmp["balisee_id"] == "Balisee_3"]
        if not b3_row.empty:
            freq_stride6 = float(b3_row["freq_fft_sam_hz"].iloc[0])

    # Interprétation
    if 0.3 <= dom_freq <= 1.5:
        interp = (
            f"La fréquence dominante ({dom_freq:.3f} Hz) se situe **dans la bande "
            f"biologique attendue (0.3–1.5 Hz)** pour *Rhizostoma pulmo*. "
            f"La fondamentale de pulsation était sous-échantillonnée à vid_stride=6 "
            f"({freq_stride6:.3f} Hz) : le signal à 5 Hz effectifs capturait une "
            f"composante basse fréquence, pas la vraie fondamentale. "
            f"SAM à 29.97 Hz révèle la pulsation réelle."
        )
    else:
        interp = (
            f"La fréquence dominante ({dom_freq:.3f} Hz) reste **hors de la bande "
            f"biologique attendue (0.3–1.5 Hz)**. "
            f"La pulsation observée ({dom_freq:.3f} Hz à vid_stride=1 vs "
            f"{freq_stride6:.3f} Hz à vid_stride=6) est effectivement lente, "
            f"compatible avec des méduses au repos ou une espèce à cycle lent. "
            f"L'hypothèse de sous-échantillonnage ne s'applique pas ici."
        )

    section = f"""
---

## Vérification fréquentielle haute résolution (B3, vid_stride=1)

Test réalisé sur le segment frames {FRAME_START}–{FRAME_END} (~{duree_s:.0f}s),
méthode SAM uniquement, vid_stride=1 (29.97 Hz natif, ~{FRAME_END - FRAME_START} points).

| Paramètre | Valeur |
|-----------|--------|
| Fréquence FFT dominante (vid_stride=1) | **{dom_freq:.3f} Hz** |
| Fréquence FFT dominante (vid_stride=6) | {freq_stride6:.3f} Hz |
| Pics détectés | {n_pics} en {duree_s:.0f}s → {freq_pics:.3f} Hz |
| Bande biologique *Rhizostoma pulmo* | 0.3–1.5 Hz |

### Interprétation

{interp}

Voir figure : `figures/pulsation_hifreq_B3.png`

*Généré par `src/inference_sam_hifreq.py` — Phase 6 vérification HF*
"""

    # Ajoute avant la ligne finale *Généré automatiquement*
    marker = "*Généré automatiquement par"
    if marker in rpt:
        rpt = rpt.replace(marker, section + "\n" + marker)
    else:
        rpt = rpt + section

    rpt_path.write_text(rpt, encoding="utf-8")
    print(f"[OK] Section HF ajoutée au rapport : {rpt_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6 — Test haute fréquence SAM sur B3 (vid_stride=1)"
    )
    parser.add_argument("--video",  default=DEFAULT_VIDEO,  type=Path)
    parser.add_argument("--detect", default=DEFAULT_DETECT, type=Path)
    parser.add_argument("--sam",    default=DEFAULT_SAM,    type=Path)
    parser.add_argument("--no-inference", action="store_true",
                        help="Passe l'inférence et utilise le CSV existant")
    args = parser.parse_args()

    if not args.no_inference:
        run_inference(args.video, args.detect, args.sam)

    result = analyze_and_plot()
    if result is not None:
        dom_freq, freq_pics, n_pics, duree_s = result
        append_report_section(dom_freq, freq_pics, n_pics, duree_s)


if __name__ == "__main__":
    main()
