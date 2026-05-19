# Phase 5 — Entraînement YOLOv8n-seg sur masques SAM
import sys
import csv
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_FILE = ROOT / "logs" / "train_seg.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

DATA_YAML  = ROOT / "data" / "seg" / "data_seg.yaml"
DETECT_CSV = ROOT / "runs" / "detect" / "yolov8n_1280_e150" / "results.csv"
REPORT_OUT = ROOT / "logs" / "phase5_report.txt"


# ─── référence détection ──────────────────────────────────────────────────────
def read_detect_best():
    with open(DETECT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    def get(row, k):
        for rk in row:
            if rk.strip() == k:
                return float(row[rk])
        raise KeyError(k)

    best_row = max(rows, key=lambda r: get(r, "metrics/mAP50-95(B)"))
    return {
        "epoch":        int(best_row["epoch"].strip()),
        "mAP50_box":    get(best_row, "metrics/mAP50(B)"),
        "mAP50_95_box": get(best_row, "metrics/mAP50-95(B)"),
        "precision":    get(best_row, "metrics/precision(B)"),
        "recall":       get(best_row, "metrics/recall(B)"),
    }


# ─── entraînement ─────────────────────────────────────────────────────────────
def train():
    from ultralytics import YOLO

    detect_ref = read_detect_best()
    log.info(
        "Référence détection (best epoch %d): mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f",
        detect_ref["epoch"], detect_ref["mAP50_box"], detect_ref["mAP50_95_box"],
        detect_ref["precision"], detect_ref["recall"],
    )

    model = YOLO("yolov8n-seg.pt")
    log.info("Lancement entraînement YOLOv8n-seg — 150 epochs, imgsz=1280")

    results = model.train(
        data=str(DATA_YAML),
        epochs=150,
        imgsz=1280,
        batch=-1,
        patience=30,
        device=0,
        project=str(ROOT / "runs" / "segment"),
        name="yolov8n_seg_1280_e150",
        close_mosaic=10,
        mixup=0.0,
        save=True,
        plots=True,
    )

    # ─── lecture des métriques finales ────────────────────────────────────────
    run_dir = Path(results.save_dir)
    seg_csv = run_dir / "results.csv"

    with open(seg_csv, newline="") as f:
        reader = csv.DictReader(f)
        seg_rows = list(reader)

    def get_seg(row, k):
        for rk in row:
            if rk.strip() == k:
                return float(row[rk])
        raise KeyError(k)

    best_seg     = max(seg_rows, key=lambda r: get_seg(r, "metrics/mAP50-95(M)"))
    last_seg     = seg_rows[-1]
    best_epoch   = int(best_seg["epoch"].strip())
    last_epoch   = int(last_seg["epoch"].strip())

    mAP50_box_seg    = get_seg(best_seg, "metrics/mAP50(B)")
    mAP50_95_box_seg = get_seg(best_seg, "metrics/mAP50-95(B)")
    mAP50_mask       = get_seg(best_seg, "metrics/mAP50(M)")
    mAP50_95_mask    = get_seg(best_seg, "metrics/mAP50-95(M)")
    prec_seg         = get_seg(best_seg, "metrics/precision(B)")
    rec_seg          = get_seg(best_seg, "metrics/recall(B)")
    val_box_loss_best = get_seg(best_seg, "val/box_loss")
    val_box_loss_last = get_seg(last_seg, "val/box_loss")

    # ─── seuils ───────────────────────────────────────────────────────────────
    stop_flag   = mAP50_mask < 0.50
    delta_map50 = detect_ref["mAP50_box"] - mAP50_box_seg
    signal_flag = delta_map50 > 0.10

    # ─── FPS ──────────────────────────────────────────────────────────────────
    speed = getattr(results, "speed", None)
    if speed and isinstance(speed, dict):
        total_ms = speed.get("preprocess", 0) + speed.get("inference", 0) + speed.get("postprocess", 0)
        fps = 1000.0 / total_ms if total_ms > 0 else None
    else:
        fps = None

    # ─── rapport ──────────────────────────────────────────────────────────────
    fps_line = f"  FPS (val, GPU 0)         : {fps:.1f}" if fps else "  FPS : n/a"
    lines = [
        "=" * 60,
        "PHASE 5 — RAPPORT ENTRAÎNEMENT YOLOv8n-seg",
        "=" * 60,
        "",
        f"Run dir : {run_dir}",
        f"Best epoch (mAP50-95 mask) : {best_epoch} / {last_epoch}",
        "",
        "── Métriques seg (best epoch) ──────────────────────────",
        f"  mAP50   box  : {mAP50_box_seg:.4f}",
        f"  mAP50-95 box : {mAP50_95_box_seg:.4f}",
        f"  mAP50   mask : {mAP50_mask:.4f}",
        f"  mAP50-95 mask: {mAP50_95_mask:.4f}",
        f"  Precision(B) : {prec_seg:.4f}",
        f"  Recall(B)    : {rec_seg:.4f}",
        "",
        "── Référence détection (best epoch) ────────────────────",
        f"  mAP50   box  : {detect_ref['mAP50_box']:.4f}",
        f"  mAP50-95 box : {detect_ref['mAP50_95_box']:.4f}",
        f"  Precision(B) : {detect_ref['precision']:.4f}",
        f"  Recall(B)    : {detect_ref['recall']:.4f}",
        "",
        "── Comparaison ─────────────────────────────────────────",
        f"  ΔmAP50 box (detect→seg) : {mAP50_box_seg - detect_ref['mAP50_box']:+.4f}",
        f"  ΔmAP50-95 box           : {mAP50_95_box_seg - detect_ref['mAP50_95_box']:+.4f}",
        "",
        "── Analyse overfitting ──────────────────────────────────",
        f"  val/box_loss best epoch  : {val_box_loss_best:.5f}",
        f"  val/box_loss last epoch  : {val_box_loss_last:.5f}",
        f"  Early stop déclenché     : {'oui' if last_epoch < 150 else 'non'} (arrêt epoch {last_epoch})",
        "",
        "── Vitesse inférence ────────────────────────────────────",
        fps_line,
        "",
        "── Seuils ───────────────────────────────────────────────",
        f"  [{'STOP' if stop_flag else 'OK  '}] mAP50 mask = {mAP50_mask:.4f}  (seuil < 0.50)",
        f"  [{'SIGNAL' if signal_flag else 'OK    '}] ΔmAP50 box vs detect = {delta_map50*100:.1f} pts  (seuil > 10 pts)",
        "",
        "=" * 60,
    ]

    report_text = "\n".join(lines)
    log.info("\n" + report_text)
    REPORT_OUT.write_text(report_text, encoding="utf-8")
    log.info("Rapport sauvegardé : %s", REPORT_OUT)

    if stop_flag:
        log.error("SEUIL STOP : mAP50 mask = %.4f < 0.50 — Phase 6 suspendue.", mAP50_mask)
    if signal_flag:
        log.warning("SIGNAL : mAP50 box a chuté de %.1f pts vs détection.", delta_map50 * 100)


if __name__ == "__main__":
    train()
