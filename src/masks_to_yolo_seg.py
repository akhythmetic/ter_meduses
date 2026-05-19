"""
Phase 4 — Masques .npz  polygones YOLO-seg.

Pour chaque frame :
  - Lit le .npz de data/seg/masks_raw/
  - Pour low_conf : clip mask & bbox avant findContours, loggue dans clipping_log.csv
  - Pour good / usable : masque tel quel
  - findContours  -> approxPolyDP adaptatif (cible 20-50 pts)
  - Normalise, écrit data/seg/labels/{split}/{frame}.txt
  - Sanity-check : n_polygones_écrits == n_bboxes_dans_label_detection

Sorties :
  data/seg/labels/{train,val}/*.txt
  data/seg/images/{train,val}/ (copie des images)
  data/seg/labels_quality.csv
  debug/clipping_log.csv
  debug/seg_label_check/ (5 visus)
"""

import csv
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
IMG_W, IMG_H = 3840, 2160
VIS_W = 1280   # largeur cible des visu debug
VIS_H = int(IMG_H * VIS_W / IMG_W)

MASKS_RAW      = REPO / "data" / "seg" / "masks_raw"
SEG_LABELS     = REPO / "data" / "seg" / "labels"
SEG_IMAGES     = REPO / "data" / "seg" / "images"
CURR_LABELS    = REPO / "data" / "current" / "labels"
CURR_IMAGES    = REPO / "data" / "current" / "images"
SUMMARY_CSV    = REPO / "logs" / "sam_generation_summary.csv"
LABELS_QUALITY = REPO / "data" / "seg" / "labels_quality.csv"
CLIPPING_LOG   = REPO / "debug" / "clipping_log.csv"
DEBUG_VIS_DIR  = REPO / "debug" / "seg_label_check"

# Couleurs debug (BGR)
COLOR_GOOD     = (0, 255, 0)    # vert
COLOR_USABLE   = (0, 255, 255)  # jaune
COLOR_LOWCONF  = (0, 165, 255)  # orange
COLOR_BBOX     = (0, 0, 255)    # rouge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(REPO / "logs" / "phase4_conversion.log",
                            mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def flag_color(flag: str):
    if flag == "good":    return COLOR_GOOD
    if flag == "usable":  return COLOR_USABLE
    return COLOR_LOWCONF


def make_bbox_mask(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    xi1, yi1 = max(0, int(x1)),     max(0, int(y1))
    xi2, yi2 = min(IMG_W, int(np.ceil(x2))), min(IMG_H, int(np.ceil(y2)))
    m[yi1:yi2, xi1:xi2] = 1
    return m


def mask_to_poly_pts(mask: np.ndarray,
                     x1: float, y1: float, x2: float, y2: float
                     ) -> list[tuple[float, float]]:
    """
    Extrait les points du polygone depuis un masque binaire.
    Retourne une liste de (x_px, y_px).
    Fallback sur le rectangle bbox si le masque est vide ou contour trop petit.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    bbox_rect = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    if not contours:
        return bbox_rect

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1:
        return bbox_rect

    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter < 1:
        return bbox_rect

    # Epsilon adaptatif — cible ~35 points (milieu [20, 50])
    eps = max(0.3, perimeter / 35.0)
    approx = cv2.approxPolyDP(contour, eps, closed=True)

    # Ajustements si hors [20, 50]
    if len(approx) > 50:
        eps = perimeter / 50.0
        approx = cv2.approxPolyDP(contour, eps, closed=True)
    if len(approx) < 6:
        eps = max(0.1, perimeter / 50.0)
        approx = cv2.approxPolyDP(contour, eps, closed=True)
    if len(approx) < 3:
        return bbox_rect

    pts = approx.reshape(-1, 2)
    return [(float(p[0]), float(p[1])) for p in pts]


# ─── Chargement données ───────────────────────────────────────────────────────

def load_summary() -> dict:
    """Index (frame_name, bbox_id) → info dict."""
    idx = {}
    float_f = {"bbox_cx_px", "bbox_cy_px", "bbox_width_px", "bbox_height_px",
               "bbox_area_px", "ratio", "iou"}
    int_f   = {"mask_area_px", "bbox_id", "frame_idx"}
    with open(SUMMARY_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["frame_name"], int(row["bbox_id"]))
            idx[key] = {
                k: (float(v) if k in float_f else int(v) if k in int_f else v)
                for k, v in row.items()
            }
    log.info(f"Summary chargé : {len(idx)} entrées")
    return idx


def collect_frames() -> list[dict]:
    """Liste de toutes les frames avec chemins."""
    frames = []
    for split in ("train", "val"):
        for npz_path in sorted(MASKS_RAW.glob("*.npz")):
            frame_name = npz_path.stem
            img_path   = CURR_IMAGES  / split / f"{frame_name}.jpg"
            det_label  = CURR_LABELS  / split / f"{frame_name}.txt"
            if not img_path.exists():
                continue
            frames.append({
                "frame_name": frame_name,
                "frame_idx":  int(frame_name.split("_")[1]),
                "split":      split,
                "npz_path":   npz_path,
                "img_path":   img_path,
                "det_label":  det_label,
                "seg_label":  SEG_LABELS / split / f"{frame_name}.txt",
            })
    return frames


# ─── Sélection frames de debug ────────────────────────────────────────────────

def select_debug_frames(frames: list[dict], summary: dict,
                        balisee_frame_names: set[str]) -> list[dict]:
    """Retourne 5 frames représentatives pour les visus."""

    # Stats par frame
    by_frame = defaultdict(lambda: {"flags": [], "split": ""})
    for (fn, _), info in summary.items():
        by_frame[fn]["flags"].append(info["quality_flag"])
        by_frame[fn]["split"] = info["split"]

    def flag_frac(fn, flag):
        d = by_frame[fn]
        n = len(d["flags"])
        return sum(1 for f in d["flags"] if f == flag) / n if n else 0

    frame_map = {f["frame_name"]: f for f in frames}

    chosen = {}

    # 1. Balisée avec max de 'good'
    bal_candidates = [
        (fn, sum(1 for fl in by_frame[fn]["flags"] if fl == "good"))
        for fn in balisee_frame_names if fn in by_frame and len(by_frame[fn]["flags"]) >= 2
    ]
    if bal_candidates:
        best_bal = max(bal_candidates, key=lambda x: x[1])[0]
        if best_bal in frame_map:
            chosen["balisee_good"] = frame_map[best_bal]

    # 2. Majoritairement usable
    usable_cands = [
        fn for fn in by_frame if flag_frac(fn, "usable") > 0.65 and len(by_frame[fn]["flags"]) >= 4
    ]
    if usable_cands:
        chosen["majority_usable"] = frame_map[sorted(usable_cands)[0]]

    # 3. Majoritairement low_conf
    lc_cands = [
        fn for fn in by_frame if flag_frac(fn, "low_conf") > 0.75 and len(by_frame[fn]["flags"]) >= 4
    ]
    if lc_cands:
        chosen["majority_lowconf"] = frame_map[sorted(lc_cands)[0]]

    # 4. Dense (plus de bboxes)
    densest = max(by_frame.items(), key=lambda x: len(x[1]["flags"]))
    if densest[0] in frame_map:
        chosen["dense"] = frame_map[densest[0]]

    # 5. Val tardif (reflets)
    val_late = [
        f for f in frames
        if f["split"] == "val" and f["frame_idx"] > 1000
    ]
    if val_late:
        chosen["reflet_val"] = max(val_late, key=lambda f: f["frame_idx"])

    log.info(f"Frames debug sélectionnées : {list(chosen.keys())}")
    return list(chosen.values())


# ─── Visualisation debug ──────────────────────────────────────────────────────

def draw_debug_frame(frame: dict, summary: dict, output_path: Path) -> None:
    img = cv2.imread(str(frame["img_path"]))
    if img is None:
        log.warning(f"Impossible de charger {frame['img_path']}")
        return

    seg_label = frame["seg_label"]
    det_label = frame["det_label"]

    # Bboxes originales (rouge)
    if det_label.exists():
        for line in det_label.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = (float(parts[k]) for k in (1, 2, 3, 4))
            x1 = int((cx - bw/2) * IMG_W); y1 = int((cy - bh/2) * IMG_H)
            x2 = int((cx + bw/2) * IMG_W); y2 = int((cy + bh/2) * IMG_H)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_BBOX, 2)

    # Polygones seg (couleur par quality_flag)
    if seg_label.exists():
        fn = frame["frame_name"]
        bbox_id = 0
        for line in seg_label.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            coords = [float(v) for v in parts[1:]]
            pts = np.array(
                [(int(coords[i] * IMG_W), int(coords[i + 1] * IMG_H))
                 for i in range(0, len(coords) - 1, 2)],
                dtype=np.int32,
            )
            key  = (fn, bbox_id)
            flag = summary.get(key, {}).get("quality_flag", "low_conf")
            color = flag_color(flag)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=2)
            # Petit label texte
            if len(pts) > 0:
                tx, ty = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                cv2.putText(img, f"{flag[0].upper()}{bbox_id}", (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            bbox_id += 1

    # Légende
    legend = (f"{frame['frame_name']} [{frame['split']}]  "
              f"G=good(green) U=usable(yellow) L=low_conf(orange) bbox=red")
    cv2.putText(img, legend, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Resize
    out = cv2.resize(img, (VIS_W, VIS_H))
    cv2.imwrite(str(output_path), out)
    log.info(f"  Debug visu : {output_path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 70)
    log.info("Phase 4 — Masques .npz  polygones YOLO-seg")
    log.info("=" * 70)

    # Préparer dossiers
    for split in ("train", "val"):
        (SEG_LABELS / split).mkdir(parents=True, exist_ok=True)
        (SEG_IMAGES / split).mkdir(parents=True, exist_ok=True)
    DEBUG_VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Chargements
    summary = load_summary()
    frames  = collect_frames()
    log.info(f"Frames à traiter : {len(frames)}")

    # Balises pour sélection debug
    balisee_frame_names: set[str] = set()
    traj_csv = REPO / "results" / "trajectories_DJI0013_final.csv"
    if traj_csv.exists():
        with open(traj_csv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("marked", "0") == "1":
                    try:
                        fi = int(float(row["frame_idx"]))
                        if 0 <= fi <= 1441:
                            balisee_frame_names.add(f"frame_{fi:05d}")
                    except ValueError:
                        pass

    # CSV files
    qf_fields   = ["frame_name", "split", "bbox_id", "quality_flag", "n_poly_pts"]
    clip_fields = ["frame_name", "bbox_id", "quality_flag",
                   "mask_area_before", "mask_area_after",
                   "pixels_clipped", "ratio_before", "ratio_after"]
    qf_fh   = open(LABELS_QUALITY, "w", newline="", encoding="utf-8")
    clip_fh = open(CLIPPING_LOG,   "w", newline="", encoding="utf-8")
    qf_writer   = csv.DictWriter(qf_fh,   fieldnames=qf_fields);   qf_writer.writeheader()
    clip_writer = csv.DictWriter(clip_fh, fieldnames=clip_fields); clip_writer.writeheader()

    # Compteurs
    total_polys    = 0
    fallback_bbox  = 0
    total_clipped  = 0
    sanity_errors  = []
    pts_counts     = []

    pbar = tqdm(frames, unit="frame", dynamic_ncols=True)
    for frame in pbar:
        fn    = frame["frame_name"]
        split = frame["split"]
        pbar.set_postfix({"frame": fn})

        # Charger .npz
        npz = np.load(str(frame["npz_path"]))
        masks    = npz["masks"]    # (N, 2160, 3840)
        bbox_ids = npz["bbox_ids"] # (N,)

        if masks.shape[0] == 0:
            frame["seg_label"].write_text("")
            continue

        polygons: list[dict] = []

        for i in range(len(bbox_ids)):
            bid = int(bbox_ids[i])
            key = (fn, bid)

            if key not in summary:
                log.warning(f"Clé absente du summary : {key}")
                continue

            info = summary[key]
            flag = info["quality_flag"]
            cx   = info["bbox_cx_px"];   cy = info["bbox_cy_px"]
            bw   = info["bbox_width_px"]; bh = info["bbox_height_px"]
            x1, y1 = cx - bw / 2, cy - bh / 2
            x2, y2 = cx + bw / 2, cy + bh / 2
            area_bbox = bw * bh

            mask = masks[i].astype(np.uint8)

            # ── Clipping low_conf ──────────────────────────────────────────
            if flag == "low_conf":
                area_before = int(mask.sum())
                bbox_m = make_bbox_mask(x1, y1, x2, y2)
                mask   = (mask & bbox_m).astype(np.uint8)
                area_after  = int(mask.sum())
                clipped = area_before - area_after
                if clipped > 0:
                    total_clipped += clipped
                    clip_writer.writerow({
                        "frame_name":       fn,
                        "bbox_id":          bid,
                        "quality_flag":     flag,
                        "mask_area_before": area_before,
                        "mask_area_after":  area_after,
                        "pixels_clipped":   clipped,
                        "ratio_before": round(area_before / area_bbox, 4) if area_bbox else 0,
                        "ratio_after":  round(area_after  / area_bbox, 4) if area_bbox else 0,
                    })

            # ── Contour + DP ──────────────────────────────────────────────
            pts = mask_to_poly_pts(mask, x1, y1, x2, y2)
            if len(pts) <= 4 and all(
                p in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)] for p in pts
            ):
                fallback_bbox += 1

            # Normaliser + clamper
            pts_norm = [
                (max(0.0, min(1.0, px / IMG_W)),
                 max(0.0, min(1.0, py / IMG_H)))
                for px, py in pts
            ]
            n_pts = len(pts_norm)
            pts_counts.append(n_pts)
            total_polys += 1

            polygons.append({"bid": bid, "flag": flag, "pts": pts_norm})
            qf_writer.writerow({
                "frame_name":  fn,
                "split":       split,
                "bbox_id":     bid,
                "quality_flag": flag,
                "n_poly_pts":  n_pts,
            })

        # ── Écriture .txt ──────────────────────────────────────────────────
        seg_label = frame["seg_label"]
        with open(seg_label, "w") as fh:
            for poly in polygons:
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly["pts"])
                fh.write(f"0 {coords}\n")

        # ── Sanity check ───────────────────────────────────────────────────
        det_label = frame["det_label"]
        if det_label.exists():
            n_det   = sum(1 for l in det_label.read_text().splitlines() if l.strip())
            n_seg   = len(polygons)
            if n_det != n_seg:
                sanity_errors.append({
                    "frame_name": fn, "n_det": n_det, "n_seg": n_seg
                })

    qf_fh.close()
    clip_fh.close()
    pbar.close()

    # ── Copie images ───────────────────────────────────────────────────────
    log.info("Copie des images vers data/seg/images/…")
    n_copied = 0
    for split in ("train", "val"):
        src_dir = CURR_IMAGES / split
        dst_dir = SEG_IMAGES  / split
        img_files = sorted(src_dir.glob("*.jpg"))
        pbar2 = tqdm(img_files, desc=f"copy {split}", unit="img", dynamic_ncols=True)
        for src in pbar2:
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                n_copied += 1
        pbar2.close()
    log.info(f"Images copiées : {n_copied}")

    # ── Sanity check résultats ─────────────────────────────────────────────
    if sanity_errors:
        log.error(f"SANITY CHECK : {len(sanity_errors)} frame(s) avec mismatch !")
        for e in sanity_errors[:10]:
            log.error(f"  {e['frame_name']}: det={e['n_det']} seg={e['n_seg']}")
    else:
        log.info("SANITY CHECK : OK — tous les polygones correspondent aux bboxes")

    # ── Debug visualisations ───────────────────────────────────────────────
    log.info("Génération des 5 visus debug…")
    debug_frames = select_debug_frames(frames, summary, balisee_frame_names)
    for i, df in enumerate(debug_frames[:5]):
        out_p = DEBUG_VIS_DIR / f"debug_{i+1:02d}_{df['frame_name']}.png"
        draw_debug_frame(df, summary, out_p)

    # ── Rapport ───────────────────────────────────────────────────────────
    import statistics
    sep = "=" * 70
    print(f"\n{sep}")
    print("RAPPORT PHASE 4")
    print(sep)
    print(f"  Polygones générés    : {total_polys}")
    print(f"  Fallbacks bbox rect  : {fallback_bbox} ({100*fallback_bbox/total_polys:.1f}%)")
    print(f"  Pixels clippés total : {total_clipped:,} (low_conf uniquement)")
    print(f"  Sanity errors        : {len(sanity_errors)}")
    if pts_counts:
        print(f"  Points/polygone : médiane={statistics.median(pts_counts):.0f} "
              f"moyenne={statistics.mean(pts_counts):.1f} "
              f"min={min(pts_counts)} max={max(pts_counts)}")
    # Count label files
    for split in ("train", "val"):
        n = len(list((SEG_LABELS / split).glob("*.txt")))
        n_img = len(list((SEG_IMAGES / split).glob("*.jpg")))
        print(f"  {split:5s}: {n} label files, {n_img} images")
    print(f"  labels_quality.csv   : {LABELS_QUALITY}")
    print(f"  clipping_log.csv     : {CLIPPING_LOG}")
    print(f"  debug visus          : {DEBUG_VIS_DIR}")
    print(sep)

    log.info("Phase 4 terminée.")


if __name__ == "__main__":
    main()
