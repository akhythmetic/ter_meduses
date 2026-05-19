"""
Phase 2 — Test SAM sur 30 méduses échantillonnées.

Vérifications intégrées :
1. Print debug dénormalisation bbox sur la 1ère bbox traitée
2. Multi-mask : masque avec score de confiance le plus élevé retenu
3. Bboxes dégénérées (w<20px ou h<20px) loggées à part, exclues des stats
"""

import csv
import logging
import random
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

IMG_W, IMG_H = 3840, 2160
DEGEN_W = 20   # px — seuil dégénéré
DEGEN_H = 20
TINY_W  = 30   # px — seuil "très petite"
SEED    = 42
N_TARGET = {"balisee": 10, "medium": 10, "tiny": 5, "reflet": 5}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(REPO / "logs" / "phase2_sam_test.log",
                            mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─── Utilitaires ──────────────────────────────────────────────────────────────

def denorm_xyxy(cx: float, cy: float, w: float, h: float, W: int, H: int):
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return float(x1), float(y1), float(x2), float(y2)


def mask_to_orig_res(mask_np: np.ndarray) -> np.ndarray:
    """Resize mask to original image resolution (nearest-neighbour)."""
    h, w = mask_np.shape[-2], mask_np.shape[-1]
    if h == IMG_H and w == IMG_W:
        return mask_np.astype(np.uint8)
    return cv2.resize(mask_np.astype(np.uint8), (IMG_W, IMG_H),
                      interpolation=cv2.INTER_NEAREST)


# ─── Chargement données ───────────────────────────────────────────────────────

def load_all_bboxes() -> list[dict]:
    rows = []
    for split in ("train", "val"):
        label_dir = REPO / "data" / "current" / "labels" / split
        img_dir   = REPO / "data" / "current" / "images"  / split
        for lf in sorted(label_dir.glob("*.txt")):
            img_path = img_dir / f"{lf.stem}.jpg"
            if not img_path.exists():
                continue
            frame_idx = int(lf.stem.split("_")[1])
            with open(lf) as fh:
                for bbox_id, line in enumerate(fh):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cx, cy, bw, bh = (float(parts[k]) for k in (1, 2, 3, 4))
                    x1, y1, x2, y2 = denorm_xyxy(cx, cy, bw, bh, IMG_W, IMG_H)
                    px_w = x2 - x1
                    px_h = y2 - y1
                    rows.append({
                        "frame_name": lf.stem,
                        "frame_idx":  frame_idx,
                        "split":      split,
                        "bbox_id":    bbox_id,
                        "img_path":   str(img_path),
                        "cx_norm": cx, "cy_norm": cy,
                        "bw_norm": bw, "bh_norm": bh,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "px_w":     px_w,
                        "px_h":     px_h,
                        "bbox_area": px_w * px_h,
                        "degenerate": px_w < DEGEN_W or px_h < DEGEN_H,
                    })
    return rows


def load_balisee_frame_ids() -> set[int]:
    csv_path = REPO / "results" / "trajectories_DJI0013_final.csv"
    ids: set[int] = set()
    if not csv_path.exists():
        log.warning("Pas de trajectoires CSV, catégorie balisee indisponible")
        return ids
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("marked", "0").strip() == "1":
                try:
                    ids.add(int(float(row["frame_idx"])))
                except ValueError:
                    pass
    log.info(f"Frames balisées dans CSV : {len(ids)}")
    return ids


# ─── Échantillonnage ──────────────────────────────────────────────────────────

def sample_30(all_bboxes: list[dict],
              balisee_frame_ids: set[int]) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)

    degen = [b for b in all_bboxes if     b["degenerate"]]
    valid = [b for b in all_bboxes if not b["degenerate"]]
    log.info(f"Dataset total : {len(all_bboxes)} bbox "
             f"({len(valid)} valides, {len(degen)} dégénérées)")

    used: set[tuple[str, int]] = set()
    samples: list[dict] = []

    def pick(pool, n, cat):
        chosen = rng.sample(pool, min(n, len(pool)))
        for b in chosen:
            b = dict(b)
            b["category"] = cat
            samples.append(b)
            used.add((b["frame_name"], b["bbox_id"]))
        log.info(f"  {cat}: {len(chosen)} sélectionnés (pool={len(pool)})")
        return chosen

    # 1. Balisées (tous splits, toutes tailles non-dégénérées)
    pool_bal = [b for b in valid if b["frame_idx"] in balisee_frame_ids]
    pick(pool_bal, N_TARGET["balisee"], "balisee")

    # 2. Très petites (w < 30px, non-dégénérées, non déjà sélectionnées)
    pool_tiny = [b for b in valid
                 if b["px_w"] < TINY_W
                 and (b["frame_name"], b["bbox_id"]) not in used]
    pick(pool_tiny, N_TARGET["tiny"], "tiny")

    # 3. Reflets / luminosité variable : ensemble val (conditions variées)
    max_idx = max(b["frame_idx"] for b in valid)
    pool_reflet = [b for b in valid
                   if b["split"] == "val"
                   and b["frame_idx"] > int(max_idx * 0.5)
                   and (b["frame_name"], b["bbox_id"]) not in used]
    pick(pool_reflet, N_TARGET["reflet"], "reflet")

    # 4. Medium (non-balisée, non-tiny, non-déjà choisi)
    pool_med = [b for b in valid
                if b["px_w"] >= TINY_W
                and b["frame_idx"] not in balisee_frame_ids
                and (b["frame_name"], b["bbox_id"]) not in used]
    pick(pool_med, N_TARGET["medium"], "medium")

    log.info(f"Total échantillons : {len(samples)}")
    return samples, degen


# ─── SAM ──────────────────────────────────────────────────────────────────────

def run_sam(sam_model, sample: dict, is_first: bool) -> dict | None:
    x1, y1, x2, y2 = sample["x1"], sample["y1"], sample["x2"], sample["y2"]

    # ── Vérification 1 : debug dénormalisation sur la 1ère bbox ──────────────
    if is_first:
        print("\n" + "=" * 65)
        print("DEBUG VÉRIFICATION 1 — Dénormalisation bbox")
        print(f"  Image            : {Path(sample['img_path']).name}")
        print(f"  Dimensions image : {IMG_W} x {IMG_H} px")
        print(f"  Valeurs brutes .txt :")
        print(f"    cx={sample['cx_norm']:.6f}  cy={sample['cy_norm']:.6f}"
              f"  w={sample['bw_norm']:.6f}  h={sample['bh_norm']:.6f}")
        print(f"  Valeurs converties xyxy pixels :")
        print(f"    x1={x1:.1f}  y1={y1:.1f}  x2={x2:.1f}  y2={y2:.1f}")
        print(f"    → bbox width={x2-x1:.1f} px   height={y2-y1:.1f} px")
        print(f"    → centre pixel estimé : ({(x1+x2)/2:.1f}, {(y1+y2)/2:.1f})")
        print("=" * 65 + "\n")

    results = sam_model(sample["img_path"],
                        bboxes=[[x1, y1, x2, y2]],
                        verbose=False)

    if not results or results[0].masks is None or len(results[0].masks) == 0:
        log.warning(f"SAM : aucun masque retourné pour "
                    f"{sample['frame_name']} bbox {sample['bbox_id']}")
        return None

    masks_data = results[0].masks.data.cpu().numpy()  # (N, H_m, W_m)
    n_masks = int(masks_data.shape[0])

    # ── Vérification 2 : masque avec le score de confiance le plus élevé ─────
    conf_score: float | str = "N/A"
    best_idx = 0

    if n_masks > 1:
        # Essai 1 : attribut .conf sur l'objet Masks (ultralytics SAM2)
        raw_conf = getattr(results[0].masks, "conf", None)
        if raw_conf is not None:
            confs = raw_conf.cpu().numpy().ravel()
            best_idx  = int(np.argmax(confs))
            conf_score = float(confs[best_idx])
            log.info(f"  Multi-mask ({n_masks}): scores={np.round(confs,3)}, "
                     f"retenu idx={best_idx} conf={conf_score:.4f}")
        else:
            # Fallback : masque avec l'aire la plus grande
            areas = [masks_data[i].sum() for i in range(n_masks)]
            best_idx  = int(np.argmax(areas))
            conf_score = "N/A_fallback_area"
            log.warning(f"  Multi-mask ({n_masks}): conf non dispo, "
                        f"fallback sur aire max (idx={best_idx})")
    else:
        # Un seul masque — essai quand même de lire le score
        raw_conf = getattr(results[0].masks, "conf", None)
        if raw_conf is not None:
            confs = raw_conf.cpu().numpy().ravel()
            conf_score = float(confs[0])

    mask_raw = masks_data[best_idx]          # (H_m, W_m)
    mask     = mask_to_orig_res(mask_raw)    # (IMG_H, IMG_W), uint8 0/1

    # ── Métriques ─────────────────────────────────────────────────────────────
    mask_area = int(mask.sum())
    bbox_area = float(sample["bbox_area"])
    ratio = mask_area / bbox_area if bbox_area > 0 else 0.0

    # IoU entre bounding rect du masque et bbox originale
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        iou = 0.0
    else:
        mx1, my1 = float(xs.min()), float(ys.min())
        mx2, my2 = float(xs.max()), float(ys.max())
        ix1 = max(x1, mx1); iy1 = max(y1, my1)
        ix2 = min(x2, mx2); iy2 = min(y2, my2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_orig = (x2 - x1) * (y2 - y1)
        area_rect = (mx2 - mx1) * (my2 - my1)
        union = area_orig + area_rect - inter
        iou = inter / union if union > 0 else 0.0

    return {
        "frame_id":        sample["frame_name"],
        "bbox_id":         sample["bbox_id"],
        "split":           sample["split"],
        "category":        sample["category"],
        "bbox_area":       round(bbox_area, 1),
        "mask_area":       mask_area,
        "ratio":           round(ratio, 4),
        "iou":             round(iou, 4),
        "conf_score":      round(conf_score, 4) if isinstance(conf_score, float) else conf_score,
        "n_masks_returned": n_masks,
        "px_w":            round(sample["px_w"], 1),
        "px_h":            round(sample["px_h"], 1),
        "degenerate":      sample["degenerate"],
        # pour la visu
        "_mask":       mask,
        "_img_path":   sample["img_path"],
        "_x1": x1, "_y1": y1, "_x2": x2, "_y2": y2,
    }


# ─── Visualisation ────────────────────────────────────────────────────────────

VIS_W = 960   # largeur d'un panneau dans l'image de debug
VIS_H = int(IMG_H * VIS_W / IMG_W)  # 540 px


def save_debug_image(res: dict, out_dir: Path) -> None:
    img  = cv2.imread(res["_img_path"])
    mask = res["_mask"]
    x1   = int(res["_x1"]); y1 = int(res["_y1"])
    x2   = int(res["_x2"]); y2 = int(res["_y2"])

    # Panneau 1 : image + bbox rouge
    p1 = img.copy()
    cv2.rectangle(p1, (x1, y1), (x2, y2), (0, 0, 255), 4)

    # Panneau 2 : image + masque cyan transparent + bbox rouge
    p2 = img.copy()
    overlay = np.zeros_like(img)
    overlay[mask > 0] = (255, 255, 0)   # cyan en BGR
    p2 = cv2.addWeighted(p2, 0.65, overlay, 0.35, 0)
    cv2.rectangle(p2, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Panneau 3 : masque seul N&B
    p3 = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    panels = [cv2.resize(p, (VIS_W, VIS_H)) for p in (p1, p2, p3)]
    combined = np.hstack(panels)

    label = (f"{res['frame_id']} | bbox={res['bbox_id']} | {res['category']} | "
             f"ratio={res['ratio']:.3f} | iou={res['iou']:.3f} | "
             f"conf={res['conf_score']} | n_masks={res['n_masks_returned']}")
    cv2.putText(combined, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    fname = f"{res['frame_id']}_b{res['bbox_id']:02d}_{res['category']}.png"
    cv2.imwrite(str(out_dir / fname), combined)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 65)
    log.info("Phase 2 — Test SAM sur 30 méduses échantillonnées")
    log.info("=" * 65)

    all_bboxes        = load_all_bboxes()
    balisee_frame_ids = load_balisee_frame_ids()
    samples, degen    = sample_30(all_bboxes, balisee_frame_ids)

    log.info(f"Bboxes dégénérées totales dans le dataset : {len(degen)}")

    # ── Charger SAM ───────────────────────────────────────────────────────────
    log.info("Chargement SAM 2.1 base (téléchargement auto si absent)…")
    from ultralytics import SAM
    sam_model = SAM("sam2.1_b.pt")
    log.info("Modèle SAM prêt.")

    # ── Inférence ─────────────────────────────────────────────────────────────
    debug_dir = REPO / "debug" / "sam_samples"
    metrics: list[dict] = []
    first = True

    for i, sample in enumerate(samples):
        log.info(f"[{i+1:02d}/30] {sample['frame_name']} bbox={sample['bbox_id']} "
                 f"cat={sample['category']} "
                 f"w={sample['px_w']:.1f}px h={sample['px_h']:.1f}px "
                 f"degen={sample['degenerate']}")

        res = run_sam(sam_model, sample, is_first=first)
        first = False

        if res is None:
            log.error(f"  → ÉCHEC, échantillon ignoré")
            continue

        log.info(f"  → ratio={res['ratio']:.3f}  iou={res['iou']:.3f}  "
                 f"conf={res['conf_score']}  n_masks={res['n_masks_returned']}")

        save_debug_image(res, debug_dir)

        # Ne stocker que les colonnes CSV (pas les données numpy)
        metrics.append({k: v for k, v in res.items() if not k.startswith("_")})

    # ── CSV métriques ─────────────────────────────────────────────────────────
    csv_fields = ["frame_id", "bbox_id", "split", "category",
                  "bbox_area", "mask_area", "ratio", "iou",
                  "conf_score", "n_masks_returned", "px_w", "px_h", "degenerate"]
    csv_path = debug_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(metrics)
    log.info(f"Métriques CSV : {csv_path}")

    # ── Stats agrégées (excluent les dégénérées) ──────────────────────────────
    valid_m = [m for m in metrics if not m["degenerate"]]
    degen_m = [m for m in metrics if     m["degenerate"]]

    ratios = [m["ratio"] for m in valid_m]
    ious   = [m["iou"]   for m in valid_m]

    susp_ratio = [m for m in valid_m if m["ratio"] < 0.2 or m["ratio"] > 1.0]
    susp_iou   = [m for m in valid_m if m["iou"]   < 0.5]

    sep = "=" * 65
    print(f"\n{sep}")
    print("RÉCAP PHASE 2 — Stats agrégées (bboxes valides, dégénérées exclues)")
    print(sep)
    print(f"  Échantillons traités  : {len(metrics)}")
    print(f"  ├─ valides (non-degen): {len(valid_m)}")
    print(f"  └─ dégénérées         : {len(degen_m)}")
    if degen_m:
        for m in degen_m:
            print(f"       {m['frame_id']} bbox{m['bbox_id']} "
                  f"w={m['px_w']}px h={m['px_h']}px "
                  f"ratio={m['ratio']:.3f} iou={m['iou']:.3f}")

    if ratios:
        print(f"\n  Ratio mask_area / bbox_area :")
        print(f"    Médiane     : {statistics.median(ratios):.3f}")
        print(f"    Moyenne     : {statistics.mean(ratios):.3f}")
        if len(ratios) > 1:
            print(f"    Écart-type  : {statistics.stdev(ratios):.3f}")
        print(f"    Min / Max   : {min(ratios):.3f} / {max(ratios):.3f}")

        print(f"\n  IoU bounding_rect_masque / bbox_originale :")
        print(f"    Médiane     : {statistics.median(ious):.3f}")
        print(f"    Moyenne     : {statistics.mean(ious):.3f}")
        if len(ious) > 1:
            print(f"    Écart-type  : {statistics.stdev(ious):.3f}")

        print(f"\n  Masques suspects (ratio<0.2 ou >1.0) : {len(susp_ratio)}")
        print(f"  Masques dérapés  (IoU<0.5)            : {len(susp_iou)}")

        if susp_ratio:
            print("  Détail suspects ratio :")
            for m in susp_ratio:
                print(f"    {m['frame_id']} bbox{m['bbox_id']:02d} "
                      f"cat={m['category']} ratio={m['ratio']:.3f} iou={m['iou']:.3f}")
        if susp_iou:
            print("  Détail dérapés IoU :")
            for m in susp_iou:
                print(f"    {m['frame_id']} bbox{m['bbox_id']:02d} "
                      f"cat={m['category']} ratio={m['ratio']:.3f} iou={m['iou']:.3f}")

    print(sep)
    log.info("Phase 2 terminée.")


if __name__ == "__main__":
    main()
