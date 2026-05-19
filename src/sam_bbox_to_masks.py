"""
Phase 3 — Génération SAM sur tout le dataset.

Pour chaque frame de data/current/images/{train,val}/ :
  - Lit le label YOLO correspondant
  - Skip les bboxes avec min(w_px, h_px) < 10 (→ logs/skipped_too_small.csv)
  - Appelle SAM2.1 avec toutes les bboxes valides d'un coup (1 encoding/frame)
  - Stocke les masques en data/seg/masks_raw/<frame>.npz (np.savez_compressed)
  - Loggue dans logs/sam_generation.log
  - Reprise sur crash : frame déjà traitée (.npz présent) = skippée
  - Sauvegarde tous les 100 frames (flush CSV)

Sorties :
  logs/sam_generation_summary.csv   — métriques par bbox
  logs/skipped_too_small.csv        — bboxes < 10px
  logs/sam_generation.log           — log détaillé
  logs/balise_signal_report.csv     — signal mask_area par balise sur le temps
"""

import csv
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

IMG_W, IMG_H  = 3840, 2160
MIN_SIDE_PX   = 10        # skip si min(w,h) < cette valeur
FLUSH_EVERY   = 100       # frames entre chaque flush CSV
TRAJ_CSV      = REPO / "results" / "trajectories_DJI0013_final.csv"

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(REPO / "logs" / "sam_generation.log",
                            mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def denorm_xyxy(cx: float, cy: float, w: float, h: float,
                W: int = IMG_W, H: int = IMG_H):
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return float(x1), float(y1), float(x2), float(y2)


def bbox_iou(x1, y1, x2, y2, mask: np.ndarray) -> float:
    """IoU entre le bounding rect du masque et la bbox originale."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    mx1, my1, mx2, my2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    ix1, iy1 = max(x1, mx1), max(y1, my1)
    ix2, iy2 = min(x2, mx2), min(y2, my2)
    inter  = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a_orig = (x2 - x1) * (y2 - y1)
    a_rect = (mx2 - mx1) * (my2 - my1)
    union  = a_orig + a_rect - inter
    return inter / union if union > 0 else 0.0


def quality_flag(bw: float, bh: float, ratio: float) -> str:
    if bw >= 20 and bh >= 20 and 0.5 <= ratio <= 1.3:
        return "good"
    if bw >= 15 and bh >= 15 and 0.4 <= ratio <= 1.5:
        return "usable"
    return "low_conf"


def mask_to_orig(mask_np: np.ndarray) -> np.ndarray:
    """Resize masque vers résolution originale (nearest-neighbour)."""
    h, w = mask_np.shape[-2], mask_np.shape[-1]
    if h == IMG_H and w == IMG_W:
        return mask_np.astype(np.uint8)
    return cv2.resize(mask_np.astype(np.uint8), (IMG_W, IMG_H),
                      interpolation=cv2.INTER_NEAREST)


# ─── Chargement frames ────────────────────────────────────────────────────────

def collect_frames() -> list[dict]:
    """Retourne la liste de toutes les frames avec leurs bboxes valides."""
    frames = []
    for split in ("train", "val"):
        label_dir = REPO / "data" / "current" / "labels" / split
        img_dir   = REPO / "data" / "current" / "images"  / split
        for lf in sorted(label_dir.glob("*.txt")):
            img_path = img_dir / f"{lf.stem}.jpg"
            if not img_path.exists():
                continue
            frame_idx = int(lf.stem.split("_")[1])
            valid_bboxes, skipped_bboxes = [], []
            for bbox_id, line in enumerate(lf.read_text().splitlines()):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = (float(parts[k]) for k in (1, 2, 3, 4))
                x1, y1, x2, y2 = denorm_xyxy(cx, cy, bw, bh)
                px_w, px_h = x2 - x1, y2 - y1
                entry = {
                    "frame_name": lf.stem,
                    "frame_idx":  frame_idx,
                    "split":      split,
                    "img_path":   str(img_path),
                    "bbox_id":    bbox_id,
                    "cx_n": cx, "cy_n": cy, "bw_n": bw, "bh_n": bh,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "px_w": px_w, "px_h": px_h,
                }
                if min(px_w, px_h) < MIN_SIDE_PX:
                    skipped_bboxes.append(entry)
                else:
                    valid_bboxes.append(entry)
            frames.append({
                "frame_name":     lf.stem,
                "frame_idx":      frame_idx,
                "split":          split,
                "img_path":       str(img_path),
                "npz_path":       str(REPO / "data" / "seg" / "masks_raw" / f"{lf.stem}.npz"),
                "valid_bboxes":   valid_bboxes,
                "skipped_bboxes": skipped_bboxes,
            })
    return frames


# ─── Balises ──────────────────────────────────────────────────────────────────

def load_balise_map() -> dict[str, list[dict]]:
    """
    Retourne {balisee_id: [{frame_idx, cx_px, cy_px, w_px, h_px}, ...]}
    pour les entrées marked=1 de trajectories_DJI0013_final.csv.
    """
    bal: dict[str, list[dict]] = {}
    if not TRAJ_CSV.exists():
        log.warning(f"Trajectoires CSV introuvable : {TRAJ_CSV}")
        return bal
    with open(TRAJ_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("marked", "0").strip() != "1":
                continue
            bid = row.get("balisee_id", "").strip()
            if not bid:
                continue
            try:
                fi = int(float(row["frame_idx"]))
            except (KeyError, ValueError):
                continue
            if fi < 0 or fi > 1441:
                continue
            bal.setdefault(bid, []).append({
                "frame_idx": fi,
                "cx_px": float(row["x_center"]),
                "cy_px": float(row["y_center"]),
                "w_px":  float(row["width"]),
                "h_px":  float(row["height"]),
            })
    log.info(f"Balises chargées : {sorted(bal.keys())}")
    return bal


# ─── Traitement SAM par frame ──────────────────────────────────────────────────

def process_frame_sam(sam_model, frame: dict, img_bgr: np.ndarray) -> list[dict]:
    """
    Appelle SAM sur toutes les bboxes valides d'une frame.
    Stratégie : batch (toutes bboxes d'un coup) avec fallback per-bbox.
    Retourne une liste de dicts métriques.
    """
    valid = frame["valid_bboxes"]
    if not valid:
        return []

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    all_xyxy = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in valid]

    # ── Tentative batch ───────────────────────────────────────────────────────
    masks_data = None
    try:
        results = sam_model(img_rgb, bboxes=all_xyxy, verbose=False)
        if results and results[0].masks is not None:
            raw = results[0].masks.data.cpu().numpy()  # (N_got, H, W)
            if raw.shape[0] == len(valid):
                masks_data = raw
            else:
                log.warning(
                    f"  Batch mismatch {frame['frame_name']}: "
                    f"attendu {len(valid)}, reçu {raw.shape[0]} — fallback per-bbox"
                )
    except Exception as exc:
        log.warning(f"  Batch SAM échoué pour {frame['frame_name']}: {exc} — fallback per-bbox")

    # ── Fallback per-bbox ──────────────────────────────────────────────────────
    if masks_data is None:
        collected = []
        for xyxy in all_xyxy:
            try:
                res = sam_model(img_rgb, bboxes=[xyxy], verbose=False)
                if res and res[0].masks is not None:
                    m = res[0].masks.data.cpu().numpy()
                    collected.append(m[0])
                else:
                    collected.append(None)
            except Exception as exc2:
                log.error(f"  Per-bbox SAM échoué bbox {xyxy}: {exc2}")
                collected.append(None)
        # Stack into array (use zeros for failed ones)
        h_out = IMG_H; w_out = IMG_W
        for m in collected:
            if m is not None:
                h_out, w_out = m.shape[-2], m.shape[-1]
                break
        masks_data = np.zeros((len(valid), h_out, w_out), dtype=np.uint8)
        for i, m in enumerate(collected):
            if m is not None:
                masks_data[i] = m.astype(np.uint8)

    # ── Calcul métriques ──────────────────────────────────────────────────────
    records = []
    stored_masks = []
    for i, bbox_info in enumerate(valid):
        raw_mask = masks_data[i]
        mask = mask_to_orig(raw_mask)

        mask_area = int(mask.sum())
        bbox_area = float(bbox_info["px_w"] * bbox_info["px_h"])
        ratio     = mask_area / bbox_area if bbox_area > 0 else 0.0
        iou       = bbox_iou(bbox_info["x1"], bbox_info["y1"],
                             bbox_info["x2"], bbox_info["y2"], mask)
        flag      = quality_flag(bbox_info["px_w"], bbox_info["px_h"], ratio)

        cx_px = (bbox_info["x1"] + bbox_info["x2"]) / 2
        cy_px = (bbox_info["y1"] + bbox_info["y2"]) / 2
        records.append({
            "path_image":      bbox_info["img_path"],
            "frame_name":      frame["frame_name"],
            "frame_idx":       frame["frame_idx"],
            "split":           frame["split"],
            "bbox_id":         bbox_info["bbox_id"],
            "bbox_cx_px":      round(cx_px, 1),
            "bbox_cy_px":      round(cy_px, 1),
            "bbox_width_px":   round(bbox_info["px_w"], 2),
            "bbox_height_px":  round(bbox_info["px_h"], 2),
            "bbox_area_px":    round(bbox_area, 1),
            "mask_area_px":    mask_area,
            "ratio":           round(ratio, 4),
            "iou":             round(iou, 4),
            "quality_flag":    flag,
        })
        stored_masks.append(mask)

    return records, stored_masks


# ─── Rapport qualité ──────────────────────────────────────────────────────────

def quality_report(summary_rows: list[dict]) -> None:
    from collections import Counter
    import statistics

    print("\n" + "=" * 70)
    print("RAPPORT QUALITÉ — Phase 3")
    print("=" * 70)

    total = len(summary_rows)
    print(f"\nTotal masques générés : {total}")

    # Distribution par quality_flag
    flags = Counter(r["quality_flag"] for r in summary_rows)
    print("\n── Distribution quality_flag ──")
    for flag in ("good", "usable", "low_conf"):
        n = flags.get(flag, 0)
        print(f"  {flag:10s}: {n:6d}  ({100*n/total:.1f}%)")

    # Distribution par bucket de taille min(w,h)
    def bucket(bw, bh):
        m = min(bw, bh)
        if m < 15: return "10-15"
        if m < 20: return "15-20"
        if m < 30: return "20-30"
        return "30+"

    buckets_data: dict[str, list] = {}
    for r in summary_rows:
        b = bucket(r["bbox_width_px"], r["bbox_height_px"])
        buckets_data.setdefault(b, []).append(r)

    print("\n── Distribution par taille min(w,h) ──")
    for b in ("10-15", "15-20", "20-30", "30+"):
        rows = buckets_data.get(b, [])
        if not rows:
            print(f"  {b:7s}: 0")
            continue
        ratios = [r["ratio"] for r in rows]
        ious   = [r["iou"]   for r in rows]
        r_med = statistics.median(ratios)
        i_med = statistics.median(ious)
        flags_b = Counter(r["quality_flag"] for r in rows)
        print(f"  {b:7s}: {len(rows):6d}  "
              f"ratio_med={r_med:.3f}  iou_med={i_med:.3f}  "
              f"good={flags_b.get('good',0)} usable={flags_b.get('usable',0)} "
              f"low_conf={flags_b.get('low_conf',0)}")

    # Stats globales ratio et IoU
    ratios_all = [r["ratio"] for r in summary_rows]
    ious_all   = [r["iou"]   for r in summary_rows]
    print("\n── Stats globales ──")
    for name, data in (("ratio", ratios_all), ("iou", ious_all)):
        p5  = sorted(data)[int(len(data)*0.05)]
        p95 = sorted(data)[int(len(data)*0.95)]
        print(f"  {name}: median={statistics.median(data):.3f}  "
              f"mean={statistics.mean(data):.3f}  "
              f"std={statistics.stdev(data):.3f}  "
              f"P5={p5:.3f}  P95={p95:.3f}")

    print("=" * 70)


def balise_signal_report(summary_rows: list[dict], balise_map: dict) -> None:
    """
    Pour chaque balise physique, matche les masques dans le summary par
    (frame_idx + distance centroïde minimale), puis affiche stats + signal.
    """
    if not balise_map:
        return

    from collections import Counter, defaultdict
    import statistics

    # Index summary par frame_name
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for r in summary_rows:
        by_frame[r["frame_name"]].append(r)

    out_rows = []
    print("\n" + "=" * 70)
    print("SIGNAL MASQUE DES BALISES PHYSIQUES")
    print("=" * 70)

    for bal_id in sorted(balise_map.keys()):
        entries = sorted(balise_map[bal_id], key=lambda e: e["frame_idx"])
        matched = []

        for e in entries:
            fname = f"frame_{e['frame_idx']:05d}"
            frame_rows = by_frame.get(fname, [])
            if not frame_rows:
                continue

            # Matching par distance euclidienne des centroïdes
            best = min(
                frame_rows,
                key=lambda r: (
                    (e["cx_px"] - float(r["bbox_cx_px"])) ** 2 +
                    (e["cy_px"] - float(r["bbox_cy_px"])) ** 2
                ),
            )
            dist = (
                (e["cx_px"] - float(best["bbox_cx_px"])) ** 2 +
                (e["cy_px"] - float(best["bbox_cy_px"])) ** 2
            ) ** 0.5

            matched.append({
                "balisee_id":    bal_id,
                "frame_idx":     e["frame_idx"],
                "mask_area_px":  int(best["mask_area_px"]),
                "ratio":         float(best["ratio"]),
                "quality_flag":  best["quality_flag"],
                "bbox_w":        float(best["bbox_width_px"]),
                "bbox_h":        float(best["bbox_height_px"]),
                "match_dist_px": round(dist, 1),
            })
            out_rows.append(matched[-1])

        if not matched:
            print(f"\n{bal_id}: aucune frame correspondante dans le dataset d'entraînement")
            continue

        flags  = Counter(m["quality_flag"] for m in matched)
        areas  = [m["mask_area_px"] for m in matched]
        dists  = [m["match_dist_px"] for m in matched]

        print(f"\n{bal_id}: {len(matched)} détections matchées")
        print(f"  quality_flag    : {dict(flags)}")
        print(f"  match_dist_px   : médiane={statistics.median(dists):.1f}px  "
              f"max={max(dists):.1f}px")
        if len(areas) > 1:
            cv_pct = 100 * statistics.stdev(areas) / statistics.mean(areas) if statistics.mean(areas) > 0 else 0
            print(f"  mask_area_px    : min={min(areas)}  max={max(areas)}  "
                  f"médiane={statistics.median(areas):.0f}  "
                  f"std={statistics.stdev(areas):.1f}  CV={cv_pct:.1f}%")
            verdict = "signal de pulsation visible ✓" if cv_pct > 10 else "signal faible — vérifier"
            print(f"  → {verdict}")
        # 10 premiers points du signal
        print(f"  Échantillon (frame_idx → mask_area_px) :")
        for m in matched[:10]:
            print(f"    {m['frame_idx']:5d} → {m['mask_area_px']:5d} px²"
                  f"  [{m['quality_flag']}]  dist={m['match_dist_px']}px")

    # Sauvegarde CSV
    out_path = REPO / "logs" / "balise_signal_report.csv"
    fields = ["balisee_id", "frame_idx", "mask_area_px", "ratio",
              "quality_flag", "bbox_w", "bbox_h", "match_dist_px"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nSignal balises sauvegardé : {out_path}")
    print("=" * 70)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import torch
    from tqdm import tqdm
    from ultralytics import SAM

    log.info("=" * 70)
    log.info("Phase 3 — Génération SAM sur tout le dataset")
    log.info("=" * 70)

    # ── Préparer dossiers ─────────────────────────────────────────────────────
    masks_dir = REPO / "data" / "seg" / "masks_raw"
    masks_dir.mkdir(parents=True, exist_ok=True)
    (REPO / "logs").mkdir(exist_ok=True)

    # ── Collecter frames ──────────────────────────────────────────────────────
    log.info("Inventaire des frames…")
    frames = collect_frames()
    n_total_bbox = sum(len(f["valid_bboxes"]) + len(f["skipped_bboxes"]) for f in frames)
    n_valid_bbox = sum(len(f["valid_bboxes"]) for f in frames)
    n_skip_bbox  = sum(len(f["skipped_bboxes"]) for f in frames)
    log.info(f"Frames : {len(frames)}  |  bbox total : {n_total_bbox}  "
             f"(valides : {n_valid_bbox}  skippées <{MIN_SIDE_PX}px : {n_skip_bbox})")

    # ── Écrire fichier skipped ────────────────────────────────────────────────
    skip_csv = REPO / "logs" / "skipped_too_small.csv"
    skip_fields = ["frame_name", "frame_idx", "split", "bbox_id",
                   "bbox_width_px", "bbox_height_px"]
    with open(skip_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=skip_fields)
        w.writeheader()
        for fr in frames:
            for b in fr["skipped_bboxes"]:
                w.writerow({
                    "frame_name":    b["frame_name"],
                    "frame_idx":     b["frame_idx"],
                    "split":         b["split"],
                    "bbox_id":       b["bbox_id"],
                    "bbox_width_px": round(b["px_w"], 2),
                    "bbox_height_px": round(b["px_h"], 2),
                })
    log.info(f"Bboxes skippées loggées dans : {skip_csv}  ({n_skip_bbox} entrées)")

    # ── Reprise sur crash : frames déjà traitées ──────────────────────────────
    done_frames = {p.stem for p in masks_dir.glob("*.npz")}
    frames_todo = [f for f in frames if f["frame_name"] not in done_frames]
    log.info(f"Déjà traitées : {len(done_frames)}  |  À traiter : {len(frames_todo)}")

    # ── Charger balises ───────────────────────────────────────────────────────
    balise_map = load_balise_map()

    # ── Charger SAM ───────────────────────────────────────────────────────────
    log.info("Chargement SAM 2.1 base…")
    sam_model = SAM("sam2.1_b.pt")
    log.info("SAM prêt.")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_csv  = REPO / "logs" / "sam_generation_summary.csv"
    summary_fields = [
        "path_image", "frame_name", "frame_idx", "split", "bbox_id",
        "bbox_cx_px", "bbox_cy_px",
        "bbox_width_px", "bbox_height_px", "bbox_area_px",
        "mask_area_px", "ratio", "iou", "quality_flag",
    ]
    # Mode "append" pour préserver les données en cas de reprise
    summary_mode = "a" if done_frames else "w"
    summary_fh   = open(summary_csv, summary_mode, newline="", encoding="utf-8")
    summary_writer = csv.DictWriter(summary_fh, fieldnames=summary_fields)
    if summary_mode == "w":
        summary_writer.writeheader()

    # ── Boucle principale ─────────────────────────────────────────────────────
    all_summary_rows: list[dict] = []
    t_start = time.time()
    n_frames_done = 0
    n_frames_empty = 0

    pbar = tqdm(frames_todo, unit="frame", dynamic_ncols=True)
    for frame in pbar:
        frame_name = frame["frame_name"]
        valid = frame["valid_bboxes"]
        pbar.set_postfix({"frame": frame_name, "bboxes": len(valid)})

        if not valid:
            n_frames_empty += 1
            # Créer un .npz vide pour marquer la frame comme traitée
            np.savez_compressed(frame["npz_path"],
                                masks=np.zeros((0, 0, 0), dtype=np.uint8),
                                bbox_ids=np.array([], dtype=np.int32))
            log.debug(f"  {frame_name}: aucune bbox valide")
            continue

        t_frame = time.time()

        # Charger l'image une seule fois
        img_bgr = cv2.imread(frame["img_path"])
        if img_bgr is None:
            log.error(f"  Impossible de lire : {frame['img_path']}")
            continue

        # SAM
        try:
            records, stored_masks = process_frame_sam(sam_model, frame, img_bgr)
        except Exception as exc:
            log.error(f"  ERREUR frame {frame_name}: {exc}")
            continue

        del img_bgr  # libère mémoire

        # Stocker .npz
        if stored_masks:
            masks_arr = np.stack(stored_masks, axis=0).astype(np.uint8)  # (N, H, W)
            bbox_ids_arr = np.array([b["bbox_id"] for b in valid], dtype=np.int32)
            np.savez_compressed(
                frame["npz_path"],
                masks=masks_arr,
                bbox_ids=bbox_ids_arr,
            )

        # Log et CSV
        dt_frame = time.time() - t_frame
        log.info(
            f"  {frame_name} [{frame['split']}] "
            f"bboxes={len(valid)} "
            f"dt={dt_frame:.2f}s "
            f"ratio_med={np.median([r['ratio'] for r in records]):.3f} "
            f"iou_med={np.median([r['iou'] for r in records]):.3f}"
        )

        for r in records:
            summary_writer.writerow(r)
            all_summary_rows.append(r)

        n_frames_done += 1

        # Flush périodique
        if n_frames_done % FLUSH_EVERY == 0:
            summary_fh.flush()
            elapsed = time.time() - t_start
            eta = elapsed / n_frames_done * (len(frames_todo) - n_frames_done)
            log.info(f"  [FLUSH] {n_frames_done}/{len(frames_todo)} frames "
                     f"— {elapsed/60:.1f}min écoulées, ETA {eta/60:.1f}min")

        # Libération mémoire GPU périodique
        if n_frames_done % 200 == 0:
            import torch
            torch.cuda.empty_cache()

    summary_fh.close()
    pbar.close()

    # ── Charger TOUT le summary (inclut les reprises éventuelles) ─────────────
    if summary_mode == "a":
        all_summary_rows = []
        float_fields = {"bbox_cx_px", "bbox_cy_px", "bbox_width_px", "bbox_height_px",
                        "bbox_area_px", "ratio", "iou"}
        int_fields   = {"mask_area_px", "bbox_id", "frame_idx"}
        with open(summary_csv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                all_summary_rows.append({
                    k: (float(v) if k in float_fields else
                        (int(v)   if k in int_fields   else v))
                    for k, v in row.items()
                })

    elapsed_total = time.time() - t_start
    n_npz  = len(list(masks_dir.glob("*.npz")))
    log.info("=" * 70)
    log.info(f"Phase 3 terminée — {elapsed_total/60:.1f} min")
    log.info(f"Frames traitées ce run : {n_frames_done}  (vides : {n_frames_empty})")
    log.info(f"Fichiers .npz total : {n_npz}")
    log.info(f"Masques dans summary : {len(all_summary_rows)}")
    log.info("=" * 70)

    if not all_summary_rows:
        log.warning("Aucune donnée dans le summary — rapport ignoré.")
        return

    # ── Rapport qualité ───────────────────────────────────────────────────────
    quality_report(all_summary_rows)

    # ── Rapport signal balises ────────────────────────────────────────────────
    balise_signal_report(all_summary_rows, balise_map)


if __name__ == "__main__":
    main()
