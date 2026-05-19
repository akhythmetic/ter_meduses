"""
Phase 6.1 — Inférence triple sur DJI_0013.

Lance UNE passe de tracking (modèle détection + ByteTrack), puis à chaque
frame traitée appelle aussi le modèle YOLOv8-seg et SAM pour obtenir les
masques de segmentation. Produit 3 CSV partageant les mêmes (frame_idx,
jelly_id) :

  results/trajectories_detection_DJI0013.csv
  results/trajectories_seg_yolovseg_DJI0013.csv
  results/trajectories_seg_sam_DJI0013.csv

Convention frame_idx :
  frame_idx = iter_count * vid_stride  (numéro de frame source)
  time_s    = frame_idx / fps_source   (fps source = 29.97)
  → Pour filtrer les 40 premières secondes : time_s >= 40

Usage :
    python src/inference_triple_video.py --video path/to/DJI_0013.MP4

    # Ou avec chemins explicites :
    python src/inference_triple_video.py \\
        --video    path/to/DJI_0013.MP4 \\
        --detect   runs/detect/yolov8n_1280_e150/weights/best.pt \\
        --seg      runs/segment/yolov8n_seg_1280_e150/weights/best.pt \\
        --sam      sam2.1_b.pt
"""

import argparse
import csv
import logging
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ─── Chemins par défaut ───────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent

DEFAULT_DETECT = REPO / "runs" / "detect" / "yolov8n_1280_e150" / "weights" / "best.pt"
DEFAULT_SEG    = REPO / "runs" / "segment" / "yolov8n_seg_1280_e150" / "weights" / "best.pt"
DEFAULT_SAM    = REPO / "sam2.1_b.pt"

OUT_DETECT = REPO / "results" / "trajectories_detection_DJI0013.csv"
OUT_SEG    = REPO / "results" / "trajectories_seg_yolovseg_DJI0013.csv"
OUT_SAM    = REPO / "results" / "trajectories_seg_sam_DJI0013.csv"
LOG_PATH   = REPO / "logs" / "phase6_inference.log"

VID_STRIDE  = 6
CONF        = 0.25
IMGSZ       = 1280
FPS_SOURCE  = 29.97

SAM_CACHE_EVERY   = 100   # flush GPU cache toutes les N frames
SAM_RELOAD_FACTOR = 2.0   # recharge SAM si temps moyen × ce facteur
TIMING_WINDOW     = 20    # nb frames pour la moyenne glissante ms/frame

# ─── Colonnes CSV ─────────────────────────────────────────────────────────────
COLS_DETECT = [
    "frame_idx", "time_s", "jelly_id",
    "x_center", "y_center", "width", "height",
    "bbox_area_px",
]
COLS_SEG = [
    "frame_idx", "time_s", "jelly_id",
    "x_center", "y_center", "width", "height",
    "bbox_area_px",
    "mask_area_yolovseg_px", "mask_perimeter_yolovseg_px",
    "matched_iou",
]
COLS_SAM = [
    "frame_idx", "time_s", "jelly_id",
    "x_center", "y_center", "width", "height",
    "bbox_area_px",
    "mask_area_sam_px", "mask_perimeter_sam_px",
]


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    return logging.getLogger("phase6")


# ─── Utilitaires géométrie ────────────────────────────────────────────────────

def compute_iou(b1, b2) -> float:
    """IoU entre deux boîtes xyxy."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def mask_metrics(mask_u8: np.ndarray) -> tuple[int, float]:
    """Retourne (aire_px, périmètre_px) d'un masque binaire uint8."""
    area = int(mask_u8.sum())
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
    return area, float(perimeter)


def resize_mask(mask_arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Redimensionne un masque (float ou bool) → uint8 à la résolution cible."""
    m = mask_arr.astype(np.uint8)
    if m.shape[0] == target_h and m.shape[1] == target_w:
        return m
    return cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


# ─── Extraction masque SAM ────────────────────────────────────────────────────

def extract_sam_mask(
    sam_results,
    target_h: int,
    target_w: int,
) -> np.ndarray | None:
    """
    Extrait le meilleur masque d'un résultat SAM mono-bbox (par score de conf,
    fallback sur aire max). Retourne un tableau uint8 (target_h, target_w).
    """
    if not sam_results or sam_results[0].masks is None:
        return None
    masks_data = sam_results[0].masks.data.cpu().numpy()  # (N, H_m, W_m)
    n = masks_data.shape[0]
    best_idx = 0
    if n > 1:
        raw_conf = getattr(sam_results[0].masks, "conf", None)
        if raw_conf is not None:
            confs = raw_conf.cpu().numpy().ravel()
            best_idx = int(np.argmax(confs))
        else:
            best_idx = int(np.argmax([masks_data[k].sum() for k in range(n)]))
    return resize_mask(masks_data[best_idx], target_h, target_w)


def extract_sam_masks_batch(
    sam_results,
    n_bboxes: int,
    target_h: int,
    target_w: int,
) -> list[np.ndarray | None]:
    """
    Extrait un masque par bbox depuis un appel SAM batché (N bboxes → N masques).

    Ultralytics SAM2 retourne (N, H, W) quand N bboxes sont fournies.
    Si le nombre de masques retournés diffère de n_bboxes (cas edge),
    on retourne None pour les bboxes sans masque.
    """
    if not sam_results or sam_results[0].masks is None:
        return [None] * n_bboxes

    masks_data = sam_results[0].masks.data.cpu().numpy()  # (M, H_m, W_m)
    n_returned = masks_data.shape[0]

    out: list[np.ndarray | None] = []
    for i in range(n_bboxes):
        if i < n_returned:
            out.append(resize_mask(masks_data[i], target_h, target_w))
        else:
            out.append(None)
    return out


# ─── Barre de progression ─────────────────────────────────────────────────────

class TimingTracker:
    """Moyenne glissante pour les temps par composant."""

    def __init__(self, window: int = TIMING_WINDOW):
        self._det  = deque(maxlen=window)
        self._seg  = deque(maxlen=window)
        self._sam  = deque(maxlen=window)

    def add(self, det_ms: float | None, seg_ms: float, sam_ms: float):
        if det_ms is not None:
            self._det.append(det_ms)
        self._seg.append(seg_ms)
        self._sam.append(sam_ms)

    def stats(self) -> dict:
        return {
            "det_ms": f"{np.mean(self._det):.0f}" if self._det else "-",
            "seg_ms": f"{np.mean(self._seg):.0f}" if self._seg else "-",
            "sam_ms": f"{np.mean(self._sam):.0f}" if self._sam else "-",
        }

    def mean_sam(self) -> float | None:
        return float(np.mean(self._sam)) if self._sam else None


# ─── Inférence principale ─────────────────────────────────────────────────────

def run(
    video_path: Path,
    detect_path: Path,
    seg_path: Path,
    sam_path: Path,
) -> None:
    from ultralytics import SAM, YOLO

    log = setup_logging()
    log.info(f"Vidéo      : {video_path}")
    log.info(f"Detect     : {detect_path}")
    log.info(f"Seg        : {seg_path}")
    log.info(f"SAM        : {sam_path}")
    log.info(f"VID_STRIDE : {VID_STRIDE}  CONF : {CONF}  IMGSZ : {IMGSZ}")

    # ── Vérifications ──────────────────────────────────────────────────────────
    for p in (video_path, detect_path, seg_path, sam_path):
        if not p.exists():
            log.error(f"Fichier introuvable : {p}")
            sys.exit(1)

    # ── Infos vidéo ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    fps_source = cap.get(cv2.CAP_PROP_FPS) or FPS_SOURCE
    orig_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_src  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    n_frames = (total_src + VID_STRIDE - 1) // VID_STRIDE
    log.info(f"Résolution : {orig_w}x{orig_h}  fps={fps_source:.3f}  "
             f"total={total_src}  à traiter={n_frames}")

    # ── Chargement modèles ─────────────────────────────────────────────────────
    log.info("Chargement modèles…")
    model_detect = YOLO(str(detect_path))
    model_seg    = YOLO(str(seg_path))
    sam_model    = SAM(str(sam_path))
    log.info("Modèles chargés.")

    # ── Préparation CSV ────────────────────────────────────────────────────────
    REPO.joinpath("results").mkdir(parents=True, exist_ok=True)
    f_detect = open(OUT_DETECT, "w", newline="", encoding="utf-8")
    f_seg    = open(OUT_SEG,    "w", newline="", encoding="utf-8")
    f_sam    = open(OUT_SAM,    "w", newline="", encoding="utf-8")

    w_detect = csv.DictWriter(f_detect, fieldnames=COLS_DETECT)
    w_seg    = csv.DictWriter(f_seg,    fieldnames=COLS_SEG)
    w_sam    = csv.DictWriter(f_sam,    fieldnames=COLS_SAM)
    for w in (w_detect, w_seg, w_sam):
        w.writeheader()

    # ── Boucle tracking ────────────────────────────────────────────────────────
    timing    = TimingTracker()
    sam_baseline: float | None = None
    n_det = n_seg_rows = n_sam_rows = 0
    n_matched_iou = 0      # pour le taux de matching seg
    n_iou_total   = 0

    t0 = time.time()

    results_gen = model_detect.track(
        source=str(video_path),
        conf=CONF,
        imgsz=IMGSZ,
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
        verbose=False,
        vid_stride=VID_STRIDE,
    )

    try:
        with tqdm(total=n_frames, desc="Phase 6.1", unit="fr",
                  dynamic_ncols=True) as pbar:
            for iter_count, result in enumerate(results_gen):

                frame_idx = iter_count * VID_STRIDE
                time_s    = round(frame_idx / fps_source, 4)
                frame     = result.orig_img   # BGR, résolution originale

                # Temps de détection (fourni par ultralytics)
                det_ms = result.speed.get("inference", None)

                # ── Aucune détection → passe la frame ────────────────────────
                if (
                    result.boxes is None
                    or result.boxes.id is None
                    or len(result.boxes) == 0
                ):
                    pbar.update(1)
                    continue

                boxes = result.boxes
                track_ids = boxes.id

                # ── YOLOseg ──────────────────────────────────────────────────
                t_seg0 = time.perf_counter()
                res_seg = model_seg.predict(
                    frame, imgsz=IMGSZ, conf=CONF, verbose=False
                )[0]
                seg_ms = (time.perf_counter() - t_seg0) * 1000

                seg_bboxes: list[tuple] = []
                seg_masks_up: list[np.ndarray] = []

                if res_seg.masks is not None and len(res_seg.boxes) > 0:
                    seg_masks_data = res_seg.masks.data.cpu().numpy()
                    for si in range(len(res_seg.boxes)):
                        sx1, sy1, sx2, sy2 = (
                            float(v) for v in res_seg.boxes.xyxy[si].cpu().numpy()
                        )
                        seg_bboxes.append((sx1, sy1, sx2, sy2))
                        seg_masks_up.append(
                            resize_mask(seg_masks_data[si], orig_h, orig_w)
                        )

                # ── Collecte des détections de la frame ──────────────────────
                n_dets = len(boxes)
                det_data = []   # liste de dicts : x1 y1 x2 y2 jid bw bh cx cy
                all_bboxes_sam = []

                for di in range(n_dets):
                    x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[di].cpu().numpy())
                    jid = int(track_ids[di].item())
                    bw  = x2 - x1
                    bh  = y2 - y1
                    cx  = x1 + bw / 2
                    cy  = y1 + bh / 2
                    det_data.append(dict(
                        jid=jid, x1=x1, y1=y1, x2=x2, y2=y2,
                        bw=bw, bh=bh, cx=cx, cy=cy,
                        bbox_area=bw * bh,
                    ))
                    all_bboxes_sam.append([x1, y1, x2, y2])

                # ── SAM : UN seul appel pour toutes les bboxes de la frame ────
                # (encode l'image une fois, N×  plus rapide qu'appels séparés)
                t_sam0 = time.perf_counter()
                try:
                    sam_res   = sam_model(frame, bboxes=all_bboxes_sam, verbose=False)
                    sam_masks = extract_sam_masks_batch(sam_res, n_dets, orig_h, orig_w)
                except Exception as exc:
                    log.warning(f"SAM frame={frame_idx}: {exc}")
                    sam_masks = [None] * n_dets
                sam_ms = (time.perf_counter() - t_sam0) * 1000

                # ── Écriture CSV par détection ────────────────────────────────
                for di, det in enumerate(det_data):
                    base_row = dict(
                        frame_idx    = frame_idx,
                        time_s       = time_s,
                        jelly_id     = det["jid"],
                        x_center     = round(det["cx"], 2),
                        y_center     = round(det["cy"], 2),
                        width        = round(det["bw"], 2),
                        height       = round(det["bh"], 2),
                        bbox_area_px = round(det["bbox_area"], 2),
                    )

                    w_detect.writerow(base_row)
                    n_det += 1

                    # Matching YOLOseg
                    best_iou      = 0.0
                    best_seg_mask = None
                    n_iou_total  += 1
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

                    for sj, (sx1, sy1, sx2, sy2) in enumerate(seg_bboxes):
                        iou = compute_iou((x1, y1, x2, y2), (sx1, sy1, sx2, sy2))
                        if iou > best_iou:
                            best_iou      = iou
                            best_seg_mask = seg_masks_up[sj]

                    if best_iou > 0.5 and best_seg_mask is not None:
                        area_seg, perim_seg = mask_metrics(best_seg_mask)
                        matched_iou = round(best_iou, 4)
                        n_matched_iou += 1
                    else:
                        area_seg = perim_seg = float("nan")
                        matched_iou = float("nan")

                    w_seg.writerow({
                        **base_row,
                        "mask_area_yolovseg_px":      area_seg,
                        "mask_perimeter_yolovseg_px": perim_seg,
                        "matched_iou":                matched_iou,
                    })
                    n_seg_rows += 1

                    # SAM (masque déjà calculé en batch)
                    mask_sam = sam_masks[di]
                    if mask_sam is not None:
                        area_sam, perim_sam = mask_metrics(mask_sam)
                    else:
                        area_sam = perim_sam = float("nan")

                    w_sam.writerow({
                        **base_row,
                        "mask_area_sam_px":      area_sam,
                        "mask_perimeter_sam_px": perim_sam,
                    })
                    n_sam_rows += 1

                # ── Mise à jour timing (par frame) ────────────────────────────
                timing.add(det_ms, seg_ms, sam_ms)

                # ── Flush GPU cache ────────────────────────────────────────────
                if iter_count > 0 and iter_count % SAM_CACHE_EVERY == 0:
                    torch.cuda.empty_cache()
                    log.info(f"GPU cache flush — iter {iter_count}")

                    # Détection de ralentissement SAM
                    mean_sam = timing.mean_sam()
                    if mean_sam is not None:
                        if sam_baseline is None:
                            sam_baseline = mean_sam
                        elif mean_sam > SAM_RELOAD_FACTOR * sam_baseline:
                            log.warning(
                                f"Ralentissement SAM : {mean_sam:.0f} ms vs "
                                f"baseline {sam_baseline:.0f} ms → rechargement"
                            )
                            del sam_model
                            torch.cuda.empty_cache()
                            sam_model = SAM(str(sam_path))
                            sam_baseline = None

                pbar.set_postfix(timing.stats())
                pbar.update(1)

                # ── Flush disque périodique ───────────────────────────────────
                if iter_count % 50 == 0:
                    for fh in (f_detect, f_seg, f_sam):
                        fh.flush()

    finally:
        for fh in (f_detect, f_seg, f_sam):
            fh.flush()
            fh.close()

    elapsed = time.time() - t0

    # ── Rapport final ──────────────────────────────────────────────────────────
    match_rate = (n_matched_iou / n_iou_total * 100) if n_iou_total else 0.0

    log.info("=" * 60)
    log.info(f"Durée totale    : {elapsed/60:.1f} min ({elapsed:.0f} s)")
    log.info(f"Frames traitées : {iter_count + 1}")
    log.info(f"Détections (detect CSV)  : {n_det}")
    log.info(f"Détections (seg CSV)     : {n_seg_rows}")
    log.info(f"Détections (SAM CSV)     : {n_sam_rows}")
    log.info(f"Matching YOLOseg IoU>0.5 : {n_matched_iou}/{n_iou_total} "
             f"({match_rate:.1f}%)")

    if match_rate < 70:
        log.warning(
            "ALERTE : taux de matching YOLOseg < 70 %  — "
            "vérifier la cohérence des deux modèles avant l'étape 6.2"
        )

    tmean = timing.stats()
    log.info(
        f"Temps/frame moyen — detect: {tmean['det_ms']} ms  "
        f"seg: {tmean['seg_ms']} ms  SAM: {tmean['sam_ms']} ms"
    )
    log.info(f"CSV detect : {OUT_DETECT}")
    log.info(f"CSV seg    : {OUT_SEG}")
    log.info(f"CSV SAM    : {OUT_SAM}")
    log.info("=" * 60)

    # ── Aperçu 5 premières lignes ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("APERÇU — 5 premières lignes de chaque CSV")
    for label, path in [
        ("Detection", OUT_DETECT),
        ("YOLOseg",   OUT_SEG),
        ("SAM",       OUT_SAM),
    ]:
        print(f"\n--- {label} ({path.name}) ---")
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 6:  # header + 5 lignes
                    break
                print(line.rstrip())
    print("=" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 6.1 — Inférence triple (détection + YOLOseg + SAM)"
    )
    parser.add_argument(
        "--video", required=True, type=Path,
        help="Chemin vers DJI_0013.MP4 (ou autre vidéo 4K)"
    )
    parser.add_argument(
        "--detect", default=DEFAULT_DETECT, type=Path,
        help=f"Modèle détection (défaut : {DEFAULT_DETECT})"
    )
    parser.add_argument(
        "--seg", default=DEFAULT_SEG, type=Path,
        help=f"Modèle YOLOv8-seg (défaut : {DEFAULT_SEG})"
    )
    parser.add_argument(
        "--sam", default=DEFAULT_SAM, type=Path,
        help=f"Modèle SAM (défaut : {DEFAULT_SAM})"
    )
    args = parser.parse_args()

    run(
        video_path   = args.video,
        detect_path  = args.detect,
        seg_path     = args.seg,
        sam_path     = args.sam,
    )
