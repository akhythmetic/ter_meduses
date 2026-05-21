"""Étape 4 — Ré-entraînement YOLOv8n sur dataset enrichi (3 sessions)."""

import sys
import logging
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_PATH = REPO / "logs" / "train_enriched.log"
LOG_PATH.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

from ultralytics import YOLO

DATA_YAML = REPO / "data/enriched_split/data_enriched.yaml"
BASE_MODEL = REPO / "yolov8n.pt"


def main():
    log.info("=== Etape 4 : Reentrainement enriched ===")
    log.info(f"Data YAML : {DATA_YAML}")
    log.info(f"Base model: {BASE_MODEL}")

    model = YOLO(str(BASE_MODEL))

    results = model.train(
        data=str(DATA_YAML),
        epochs=150,
        imgsz=1280,
        batch=-1,
        device=0,
        project=str(REPO / "runs/detect"),
        name="yolov8n_1280_e150_enriched",
        patience=30,
        close_mosaic=10,
        save=True,
        plots=True,
        exist_ok=True,
    )

    # Extract val metrics at best epoch
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=1280,
        conf=0.25,
        split="val",
        plots=False,
        verbose=False,
    )

    log.info("=== Metriques val (enriched) ===")
    log.info(f"mAP50      : {metrics.box.map50:.4f}")
    log.info(f"mAP50-95   : {metrics.box.map:.4f}")
    log.info(f"Precision  : {metrics.box.mp:.4f}")
    log.info(f"Recall     : {metrics.box.mr:.4f}")
    log.info("=== Entrainement termine ===")

    print("\n" + "=" * 50)
    print("METRIQUES VAL — Modele enrichi")
    print("=" * 50)
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
