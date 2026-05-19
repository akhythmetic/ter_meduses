"""
Phase 6.2 — Vidéo démo annotée (3 signaux superposés).

Segment : 133.0s – 163.0s  (5 balisées présentes simultanément).
Sortie  : results/DJI0013_three_signals_demo.mp4  (1920×1080, 29.97fps)

Overlays :
  Rouge   = bboxes détection  (tracker)
  Cyan    = contours YOLOv8-seg mask
  Magenta = contours SAM mask
  Jaune   = label "Balisée #X" pour les 5 balisées physiques

Stratégie frame :
  - vid_stride=6 → une frame analysée (détect+seg+SAM) toutes les 6 frames source
  - L'overlay de la frame analysée est maintenu ("hold") sur les 5 frames source
    suivantes → video fluide à 29.97 fps sans re-inférence inutile
"""

import csv
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent

# ─── Chemins ──────────────────────────────────────────────────────────────────
DEFAULT_VIDEO  = (REPO / "TER analyse videos meduses"
                  / "2025_06_30" / "DJI_0013.MP4")
DEFAULT_SEG    = (REPO / "runs" / "segment"
                  / "yolov8n_seg_1280_e150" / "weights" / "best.pt")
DEFAULT_SAM    = REPO / "sam2.1_b.pt"
DEFAULT_OUTPUT = REPO / "results" / "DJI0013_three_signals_demo.mp4"
DET_CSV        = REPO / "results" / "trajectories_detection_DJI0013.csv"
LOG_PATH       = REPO / "logs" / "phase6_render.log"

# ─── Paramètres segment ───────────────────────────────────────────────────────
SEG_START_S = 133.0
SEG_END_S   = 163.0
FPS_SOURCE  = 29.97
VID_STRIDE  = 6
OUT_W, OUT_H = 1920, 1080
SCALE        = OUT_W / 3840   # 0.5  (4K → 1080p)
CONF         = 0.25
IMGSZ        = 1280

# ─── Balisées ─────────────────────────────────────────────────────────────────
BALISEE_IDS = {
    1: [767, 936, 946, 958, 1165, 1217],
    2: [771, 1220, 1371],
    3: [764],
    4: [787, 863, 942, 981, 1038, 1193, 1257, 1344],
    5: [847, 945, 1178, 1237, 1288],
}
JID_TO_BAL: dict[int, int] = {}
for _bnum, _ids in BALISEE_IDS.items():
    for _jid in _ids:
        JID_TO_BAL[_jid] = _bnum

# ─── Couleurs BGR ─────────────────────────────────────────────────────────────
C_RED    = (0,   0,   255)
C_CYAN   = (255, 255,   0)
C_MAG    = (255,   0, 255)
C_YELLOW = (0,   255, 255)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,     0,   0)


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
    return logging.getLogger("phase6_render")


# ─── Chargement CSV détection ─────────────────────────────────────────────────

def load_det_csv() -> dict[int, list[dict]]:
    det: dict[int, list[dict]] = {}
    with open(DET_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fidx = int(row["frame_idx"])
            cx   = float(row["x_center"])
            cy   = float(row["y_center"])
            w    = float(row["width"])
            h    = float(row["height"])
            det.setdefault(fidx, []).append({
                "jid": int(row["jelly_id"]),
                "x1": cx - w / 2,  "y1": cy - h / 2,
                "x2": cx + w / 2,  "y2": cy + h / 2,
            })
    return det


# ─── Utilitaires géométrie / masques ─────────────────────────────────────────

def iou(b1: tuple, b2: tuple) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0., ix2 - ix1) * max(0., iy2 - iy1)
    if inter == 0:
        return 0.
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.


def mask_to_contours(mask_arr: np.ndarray) -> list[np.ndarray]:
    """Resize mask to output resolution (1080p) et retourne les contours."""
    m = cv2.resize(mask_arr.astype(np.uint8), (OUT_W, OUT_H),
                   interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


# ─── Traitement d'une frame analysée ─────────────────────────────────────────

def process_frame(
    frame_bgr: np.ndarray,
    frame_idx: int,
    det_by_frame: dict,
    model_seg,
    sam_model,
) -> dict[int, dict]:
    """
    Lance YOLOseg + SAM sur une frame analysée.
    Retourne : {jid: {'bbox_1080', 'yolo_contours', 'sam_contours', 'balisee_num'}}
    """
    dets = det_by_frame.get(frame_idx, [])
    if not dets:
        return {}

    result_seg = model_seg.predict(frame_bgr, imgsz=IMGSZ, conf=CONF, verbose=False)[0]

    seg_bboxes: list[tuple] = []
    seg_masks:  np.ndarray | None = None
    if result_seg.masks is not None and len(result_seg.boxes) > 0:
        seg_bboxes = [tuple(float(v) for v in row)
                      for row in result_seg.boxes.xyxy.cpu().numpy()]
        seg_masks  = result_seg.masks.data.cpu().numpy()   # (N, H_seg, W_seg)

    all_bboxes = [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets]
    try:
        sam_res  = sam_model(frame_bgr, bboxes=all_bboxes, verbose=False)
        sam_data = (sam_res[0].masks.data.cpu().numpy()
                    if sam_res and sam_res[0].masks is not None
                    else None)
    except Exception:
        sam_data = None

    overlay: dict[int, dict] = {}
    for di, det in enumerate(dets):
        jid = det["jid"]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

        # Bbox scaled to 1080p
        bbox_1080 = (int(x1 * SCALE), int(y1 * SCALE),
                     int(x2 * SCALE), int(y2 * SCALE))

        # YOLOseg — matching IoU > 0.5
        yolo_cnt = None
        best_iou, best_idx = 0., -1
        for sj, sbbox in enumerate(seg_bboxes):
            v = iou((x1, y1, x2, y2), sbbox)
            if v > best_iou:
                best_iou, best_idx = v, sj
        if best_iou > 0.5 and seg_masks is not None and best_idx < len(seg_masks):
            yolo_cnt = mask_to_contours(seg_masks[best_idx])

        # SAM
        sam_cnt = None
        if (sam_data is not None
                and sam_data.ndim == 3
                and di < sam_data.shape[0]):
            sam_cnt = mask_to_contours(sam_data[di])

        overlay[jid] = {
            "bbox_1080":     bbox_1080,
            "yolo_contours": yolo_cnt,
            "sam_contours":  sam_cnt,
            "balisee_num":   JID_TO_BAL.get(jid),
        }

    return overlay


# ─── Dessin sur la frame 1080p ────────────────────────────────────────────────

def draw_transparent_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple,
    alpha: float = 0.55,
) -> None:
    """Fond semi-transparent sur img (in-place)."""
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_frame(
    frame_1080: np.ndarray,
    overlay: dict,
    elapsed_s: float,
    total_s: float,
) -> np.ndarray:
    out = frame_1080.copy()
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    AA   = cv2.LINE_AA

    for jid, data in overlay.items():
        x1, y1, x2, y2 = data["bbox_1080"]
        bnum = data["balisee_num"]

        # Contours YOLOseg (cyan) — dessinés en premier (sous la bbox)
        if data["yolo_contours"]:
            cv2.drawContours(out, data["yolo_contours"], -1, C_CYAN, 2, AA)

        # Contours SAM (magenta)
        if data["sam_contours"]:
            cv2.drawContours(out, data["sam_contours"], -1, C_MAG, 2, AA)

        # Bbox détection (rouge)
        cv2.rectangle(out, (x1, y1), (x2, y2), C_RED, 2)

        # Label jelly_id — fond rouge semi-transparent
        lbl = str(jid)
        (lw, lh), _ = cv2.getTextSize(lbl, FONT, 0.42, 1)
        lx1, ly1 = x1, max(y1 - lh - 6, 0)
        draw_transparent_rect(out, (lx1, ly1), (lx1 + lw + 6, ly1 + lh + 6), C_RED)
        cv2.putText(out, lbl, (lx1 + 3, ly1 + lh + 2), FONT, 0.42, C_WHITE, 1, AA)

        # Label balisée (jaune, plus gros) au-dessus du label jid
        if bnum is not None:
            bal_txt = f"Balisee #{bnum}"
            (bw, bh), _ = cv2.getTextSize(bal_txt, FONT, 0.8, 2)
            by = max(ly1 - bh - 4, bh + 2)
            cv2.putText(out, bal_txt, (x1, by), FONT, 0.8, C_YELLOW, 2, AA)

    # ── Compteur temps (haut-gauche) ────────────────────────────────────────
    elapsed_s = max(0., elapsed_s)
    ts = (f"{int(elapsed_s//60):02d}:{elapsed_s%60:04.1f}"
          f" / {int(total_s//60):02d}:{total_s%60:04.1f}")
    (tw, th), _ = cv2.getTextSize(ts, FONT, 0.7, 2)
    pad = 8
    draw_transparent_rect(out, (4, 4), (4 + tw + 2*pad, 4 + th + 2*pad), C_BLACK, 0.60)
    cv2.putText(out, ts, (4 + pad, 4 + pad + th), FONT, 0.7, C_WHITE, 2, AA)

    # ── Légende (bas-gauche) ────────────────────────────────────────────────
    parts = [
        ("Rouge=det bbox", C_RED),
        ("  |  Cyan=YOLOv8-seg mask", C_CYAN),
        ("  |  Magenta=SAM mask", C_MAG),
    ]
    total_w = sum(cv2.getTextSize(t, FONT, 0.5, 1)[0][0] for t, _ in parts)
    ly_base = OUT_H - 10
    _, lh_leg = cv2.getTextSize("A", FONT, 0.5, 1)[0]
    draw_transparent_rect(
        out,
        (4, ly_base - lh_leg - 8),
        (4 + total_w + 12, ly_base + 4),
        C_BLACK, 0.60,
    )
    x_cur = 10
    for txt, col in parts:
        cv2.putText(out, txt, (x_cur, ly_base), FONT, 0.5, col, 1, AA)
        x_cur += cv2.getTextSize(txt, FONT, 0.5, 1)[0][0]

    return out


# ─── Rendu principal ──────────────────────────────────────────────────────────

def run(
    video_path: Path,
    seg_path:   Path,
    sam_path:   Path,
    output_path: Path,
) -> None:
    from ultralytics import SAM, YOLO

    log = setup_logging()
    log.info(f"Segment : {SEG_START_S}s – {SEG_END_S}s")
    log.info(f"Vidéo   : {video_path}")
    log.info(f"Sortie  : {output_path}")

    for p in (video_path, seg_path, sam_path, DET_CSV):
        if not p.exists():
            log.error(f"Fichier introuvable : {p}")
            sys.exit(1)

    # ── Infos vidéo ────────────────────────────────────────────────────────────
    cap      = cv2.VideoCapture(str(video_path))
    fps_src  = cap.get(cv2.CAP_PROP_FPS) or FPS_SOURCE
    orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Alignement start sur multiple de VID_STRIDE
    start_src_raw = int(SEG_START_S * fps_src)
    start_src     = (start_src_raw // VID_STRIDE) * VID_STRIDE
    n_src_frames  = int((SEG_END_S - SEG_START_S) * fps_src)
    end_src       = min(start_src + n_src_frames, n_total)
    duration_s    = (end_src - start_src) / fps_src

    log.info(f"Frames source : {start_src} → {end_src}  ({end_src - start_src} frames, {duration_s:.1f}s)")
    log.info(f"Résolution source : {orig_w}×{orig_h}  →  sortie {OUT_W}×{OUT_H}")

    # ── Modèles ─────────────────────────────────────────────────────────────────
    log.info("Chargement modèles…")
    model_seg = YOLO(str(seg_path))
    sam_model = SAM(str(sam_path))
    log.info("Modèles chargés.")

    det_by_frame = load_det_csv()
    log.info(f"CSV détection : {len(det_by_frame)} frames avec détections")

    # ── Writer vidéo ───────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_src, (OUT_W, OUT_H))
    if not writer.isOpened():
        log.error("Impossible d'ouvrir le VideoWriter")
        sys.exit(1)

    # ── Seek + lecture ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_src)
    actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    log.info(f"Seek → {start_src}  (position réelle après seek : {actual_pos})")

    current_overlay: dict = {}
    n_analyzed = 0
    t0 = time.time()

    n_to_render = end_src - start_src
    with tqdm(total=n_to_render, desc="Rendu 6.2", unit="fr", dynamic_ncols=True) as pbar:
        for i in range(n_to_render):
            src_pos = start_src + i
            ret, frame = cap.read()
            if not ret:
                log.warning(f"Lecture échouée à frame source {src_pos}")
                break

            # Frame analysée → inférence + mise à jour overlay
            if src_pos % VID_STRIDE == 0:
                current_overlay = process_frame(
                    frame, src_pos, det_by_frame, model_seg, sam_model
                )
                n_analyzed += 1

            # Resize frame 4K → 1080p
            frame_1080 = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

            # Dessin des overlays
            elapsed_in_seg = i / fps_src
            annotated = draw_frame(frame_1080, current_overlay, elapsed_in_seg, duration_s)

            writer.write(annotated)
            pbar.update(1)

    cap.release()
    writer.release()

    # ── Rapport final ───────────────────────────────────────────────────────────
    elapsed   = time.time() - t0
    file_mb   = output_path.stat().st_size / 1e6 if output_path.exists() else 0

    log.info("=" * 60)
    log.info(f"Durée rendu        : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info(f"Frames rendues     : {i + 1}")
    log.info(f"Frames analysées   : {n_analyzed}  (inférence seg+SAM)")
    log.info(f"Segment            : {start_src/fps_src:.1f}s – {end_src/fps_src:.1f}s")
    log.info(f"Fichier sortie     : {output_path}  ({file_mb:.1f} MB)")
    log.info("=" * 60)

    print(f"\n[OK] Rendu 6.2 terminé en {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"[OK] {output_path.name}  {file_mb:.1f} MB")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6.2 — Rendu vidéo démo")
    parser.add_argument("--video",  default=DEFAULT_VIDEO,  type=Path)
    parser.add_argument("--seg",    default=DEFAULT_SEG,    type=Path)
    parser.add_argument("--sam",    default=DEFAULT_SAM,    type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    args = parser.parse_args()
    run(args.video, args.seg, args.sam, args.output)
