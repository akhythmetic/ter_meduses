"""
Mise à jour de windows_selected.csv :
  1. Ajoute W16 controle_pur (coin top-left, 0 detection historique)
  2. Relance SAM a vid_stride=1 sur chaque fenetre (10s = 300 frames)
  3. Ajoute colonnes n_pulsations_modele_stride6/stride1 et freq correspondantes

Usage :
    python src/update_windows_stride1.py --video "TER analyse videos meduses/2025_06_30/DJI_0013.MP4"
    python src/update_windows_stride1.py --no-inference  # analyse seule
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.ndimage import median_filter
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent

DEFAULT_VIDEO  = REPO / "TER analyse videos meduses" / "2025_06_30" / "DJI_0013.MP4"
DEFAULT_DETECT = REPO / "runs" / "detect" / "yolov8n_1280_e150" / "weights" / "best.pt"
DEFAULT_SAM    = REPO / "sam2.1_b.pt"

CSV_IN  = REPO / "results" / "windows_selected.csv"
CSV_OUT = REPO / "results" / "windows_selected.csv"   # mise à jour in-place

FPS_SRC    = 29.97
VID_STRIDE = 6
EFF_FPS_6  = FPS_SRC / VID_STRIDE   # ~4.995 Hz
EFF_FPS_1  = FPS_SRC                 # 29.97 Hz
WIN_FRAMES = 300                     # 10s × 29.97 fps
IMGSZ      = 1280
CONF       = 0.25
TOL_SPATIAL = 200                    # ±px autour du centre balisee

# W16 : coin top-left, 0 detection historique sur toute la video
W16_CX      = 240
W16_CY      = 270
W16_T_START = 140.0   # choisi en milieu de video (drone stable)
W16_FRAME_START = int(W16_T_START * FPS_SRC)

BALISEES: dict[str, list[int]] = {
    "Balisee_1": [767, 936, 946, 958, 1165, 1217],
    "Balisee_2": [771, 1220, 1371],
    "Balisee_3": [764],
    "Balisee_4": [787, 863, 942, 981, 1038, 1193, 1257, 1344],
    "Balisee_5": [847, 945, 1178, 1237, 1288],
}


# ─── Utilitaires ──────────────────────────────────────────────────────────────

def rolling_median(arr: np.ndarray, k: int = 3) -> np.ndarray:
    if len(arr) < k:
        return arr.copy()
    return median_filter(arr.astype(float), size=k, mode="nearest")


def count_peaks_signal(areas: list[float], eff_fps: float) -> tuple[int, float]:
    """find_peaks sur un signal d'aires, retourne (n_pics, freq_hz)."""
    v = np.array([a for a in areas if np.isfinite(a) and a > 0], dtype=float)
    if len(v) < 6:
        return 0, 0.0
    v_sm = rolling_median(v, k=3)
    med  = np.median(v_sm)
    std  = np.std(v_sm)
    if med == 0 or std == 0:
        return 0, 0.0
    min_dist = max(int(eff_fps * 0.5), 1)
    peaks, _ = sp_signal.find_peaks(
        v_sm,
        height=med + 0.3 * std,
        distance=min_dist,
        prominence=max(0.15 * std, 1.0),
    )
    n   = len(peaks)
    dur = len(v) / eff_fps
    return n, round(n / dur, 4) if dur > 0 else 0.0


def mask_area(mask_arr: np.ndarray, h: int, w: int) -> int:
    m = mask_arr.astype(np.uint8)
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return int(m.sum())


# ─── Ajout W16 ────────────────────────────────────────────────────────────────

def add_w16(df: pd.DataFrame) -> pd.DataFrame:
    if "W16" in df["window_id"].values:
        print("  W16 deja present, skip.")
        return df

    frame_end  = W16_FRAME_START + WIN_FRAMES - VID_STRIDE
    t_start    = round(W16_FRAME_START / FPS_SRC, 3)
    t_end      = round(frame_end       / FPS_SRC, 3)

    w16 = {
        "window_id":           "W16",
        "balisee_physique":    "controle_pur",
        "jelly_id":            "aucun",
        "frame_start":         W16_FRAME_START,
        "frame_end":           frame_end,
        "time_start_s":        t_start,
        "time_end_s":          t_end,
        "duree_s":             round(t_end - t_start, 2),
        "x_center_moyen":      W16_CX,
        "y_center_moyen":      W16_CY,
        "taille_moyenne_px":   0.0,
        "coverage":            0.0,    # sera recalcule apres inference
        "n_pulsations_modele": 0,
        "freq_modele_hz":      0.0,
    }
    row = pd.DataFrame([w16])
    return pd.concat([df, row], ignore_index=True)


# ─── Inference stride-1 par fenetre ───────────────────────────────────────────

def run_window_stride1(
    cap: cv2.VideoCapture,
    orig_h: int, orig_w: int,
    model_det,
    sam_model,
    frame_start: int,
    cx: float, cy: float,
    desc: str = "",
) -> tuple[int, float, int]:
    """
    Traite les 300 frames [frame_start, frame_start+299] a vid_stride=1.
    Retourne (n_pics, freq_hz, n_detected).
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    areas: list[float] = []
    n_det = 0

    for _ in tqdm(range(WIN_FRAMES), desc=desc, leave=False, unit="fr"):
        ret, frame = cap.read()
        if not ret:
            break

        res = model_det.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            areas.append(float("nan"))
            continue

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()

        # Filtre spatial : detections proches du centre balisee
        candidates = []
        for bi in range(len(res.boxes)):
            x1, y1, x2, y2 = boxes_xyxy[bi]
            bcx = (x1 + x2) / 2
            bcy = (y1 + y2) / 2
            if abs(bcx - cx) <= TOL_SPATIAL and abs(bcy - cy) <= TOL_SPATIAL:
                candidates.append([x1, y1, x2, y2])

        if not candidates:
            areas.append(float("nan"))
            continue

        # SAM sur les candidats
        try:
            sam_res = sam_model(frame, bboxes=candidates, verbose=False)
            if sam_res and sam_res[0].masks is not None:
                masks_data = sam_res[0].masks.data.cpu().numpy()
                # Garde le masque avec la plus grande aire
                best_area = max(
                    mask_area(masks_data[mi], orig_h, orig_w)
                    for mi in range(masks_data.shape[0])
                )
                areas.append(float(best_area))
                n_det += 1
            else:
                areas.append(float("nan"))
        except Exception as e:
            areas.append(float("nan"))

    n_pics, freq = count_peaks_signal(areas, EFF_FPS_1)
    return n_pics, freq, n_det


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(video_path: Path, detect_path: Path, sam_path: Path,
         no_inference: bool) -> None:

    # Charge CSV
    print(f"Lecture {CSV_IN.name}...")
    df = pd.read_csv(CSV_IN)

    # Renomme les colonnes stride-6 existantes si besoin
    rename_map = {
        "n_pulsations_modele": "n_pulsations_modele_stride6",
        "freq_modele_hz":      "freq_modele_stride6_hz",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Ajoute colonnes stride-1 si absentes
    for col, default in [
        ("n_pulsations_modele_stride1", -1),
        ("freq_modele_stride1_hz",      float("nan")),
        ("n_detected_stride1",          -1),
    ]:
        if col not in df.columns:
            df[col] = default

    # Ajoute W16
    print("Ajout W16 controle_pur...")
    df = add_w16(df)

    print(f"  {len(df)} fenetres au total.")

    if no_inference:
        print("[--no-inference] Sauvegarde directe sans relance SAM.")
        df.to_csv(CSV_OUT, index=False)
        print(f"[OK] {CSV_OUT}")
        return

    # Verification chemins
    for p in (video_path, detect_path, sam_path):
        if not p.exists():
            print(f"ERREUR : {p}", file=sys.stderr); sys.exit(1)

    print("\nChargement modeles...")
    from ultralytics import SAM, YOLO
    model_det = YOLO(str(detect_path))
    sam_model = SAM(str(sam_path))
    print("Modeles charges.")

    cap = cv2.VideoCapture(str(video_path))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video : {orig_w}x{orig_h}")

    # Trie par frame_start pour seek sequentiel (optimal)
    df_sorted = df.sort_values("frame_start").reset_index(drop=True)

    print(f"\n=== Inference stride-1 sur {len(df_sorted)} fenetres ===")
    for idx, row in df_sorted.iterrows():
        wid  = row["window_id"]
        bp   = row["balisee_physique"]
        cx   = float(row["x_center_moyen"])
        cy   = float(row["y_center_moyen"])
        fs   = int(row["frame_start"])
        desc = f"{wid} {bp} ({row['time_start_s']:.0f}s)"

        # Skip si deja calcule
        _v = row.get("n_pulsations_modele_stride1", -1)
        if pd.notna(_v) and int(_v) >= 0:
            print(f"  {wid} : deja calcule, skip.")
            continue

        print(f"\n  {desc}")
        n_pics, freq, n_det = run_window_stride1(
            cap, orig_h, orig_w, model_det, sam_model,
            fs, cx, cy, desc=desc,
        )

        # Met a jour dans df original (par window_id)
        mask = df["window_id"] == wid
        df.loc[mask, "n_pulsations_modele_stride1"] = n_pics
        df.loc[mask, "freq_modele_stride1_hz"]      = freq
        df.loc[mask, "n_detected_stride1"]          = n_det

        print(f"    -> n_pics={n_pics}  freq={freq:.3f}Hz  n_det={n_det}")

        # Flush CSV apres chaque fenetre (resilience si interruption)
        df.to_csv(CSV_OUT, index=False)

    cap.release()

    # Sauvegarde finale
    df.to_csv(CSV_OUT, index=False)
    print(f"\n[OK] CSV mis a jour : {CSV_OUT}")

    # Affichage recap
    cols_show = [
        "window_id", "balisee_physique", "time_start_s",
        "n_pulsations_modele_stride6",
        "n_pulsations_modele_stride1",
        "freq_modele_stride6_hz",
        "freq_modele_stride1_hz",
    ]
    cols_show = [c for c in cols_show if c in df.columns]
    print("\n=== Recapitulatif final ===")
    print(df[cols_show].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",        default=DEFAULT_VIDEO,  type=Path)
    parser.add_argument("--detect",       default=DEFAULT_DETECT, type=Path)
    parser.add_argument("--sam",          default=DEFAULT_SAM,    type=Path)
    parser.add_argument("--no-inference", action="store_true")
    args = parser.parse_args()

    main(args.video, args.detect, args.sam, args.no_inference)
