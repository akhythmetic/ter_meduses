"""
Entraînement YOLO méduses — paramétrable via CLI ou preset.

Usage :
    python src/train.py                             # baseline
    python src/train.py --preset v8s_640
    python src/train.py --preset baseline --dataset enriched
    python src/train.py --model yolov8n.pt --imgsz 1280 --epochs 100 --name mon_run
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ACTIVE_DATASET, DATASET_DIRS, EXPERIMENTS, RUNS_DIR
from utils import append_experiment, fmt_seconds, read_results_csv, setup_dirs, write_data_yaml


def check_gpu() -> None:
    if not torch.cuda.is_available():
        print("✗  CUDA non disponible — entraînement sur CPU (extrêmement lent).")
        print("   Vérifie : version PyTorch, drivers NVIDIA, et que CUDA est installé.")
        sys.exit(1)
    gpu  = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓  GPU : {gpu}  ({vram:.1f} GB VRAM)")


def train(preset: str = "baseline", dataset: str = ACTIVE_DATASET, **overrides) -> None:
    check_gpu()

    dataset_dir = DATASET_DIRS.get(dataset)
    if dataset_dir is None or not dataset_dir.exists():
        print(f"✗  Dataset '{dataset}' introuvable : {dataset_dir}")
        sys.exit(1)

    setup_dirs(RUNS_DIR)

    # Fusionner preset + overrides CLI
    params = dict(EXPERIMENTS.get(preset, EXPERIMENTS["baseline"]))
    params.update({k: v for k, v in overrides.items() if v is not None})

    model_name = params.pop("model", "yolov8n.pt")
    run_name   = params.get("name", f"run_{preset}")

    # data.yaml corrigé (chemins absolus Windows)
    yaml_path = write_data_yaml(dataset_dir)
    print(f"✓  data.yaml généré : {yaml_path}")

    from ultralytics import YOLO
    model = YOLO(model_name)

    print(f"\n── Lancement de l'entraînement ──────────────────────────")
    print(f"   Run      : {run_name}")
    print(f"   Modèle   : {model_name}")
    print(f"   Dataset  : {dataset}  ({dataset_dir})")
    print(f"   imgsz    : {params.get('imgsz')}  |  epochs : {params.get('epochs')}  "
          f"|  batch : {params.get('batch')}  |  device : {params.get('device')}")
    print(f"──────────────────────────────────────────────────────────\n")

    t0 = time.time()
    model.train(
        data    = str(yaml_path),
        project = str(RUNS_DIR / "detect"),
        **params,
    )
    elapsed = time.time() - t0
    print(f"\n✓  Entraînement terminé en {fmt_seconds(elapsed)}")

    # Lire les métriques depuis results.csv (dernière epoch)
    run_dir = RUNS_DIR / "detect" / run_name
    raw     = read_results_csv(run_dir)

    mAP50   = float(raw.get("metrics/mAP50(B)",    0) or 0)
    mAP5095 = float(raw.get("metrics/mAP50-95(B)", 0) or 0)
    prec    = float(raw.get("metrics/precision(B)", 0) or 0)
    rec     = float(raw.get("metrics/recall(B)",    0) or 0)
    f1      = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"\n── Métriques (dernière epoch) ───────────────────────────")
    print(f"   mAP50     = {mAP50:.4f}   (cible baseline ≈ 0.715)")
    print(f"   mAP50-95  = {mAP5095:.4f}")
    print(f"   Precision = {prec:.4f}   (cible baseline ≈ 0.65)")
    print(f"   Recall    = {rec:.4f}   (cible baseline ≈ 0.57)")
    print(f"   F1        = {f1:.4f}")
    print(f"──────────────────────────────────────────────────────────")
    print(f"   Poids best.pt : {run_dir / 'weights' / 'best.pt'}")

    append_experiment({
        "date":         datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "run_name":     run_name,
        "dataset":      dataset,
        "model":        model_name,
        "imgsz":        params.get("imgsz", 640),
        "epochs":       params.get("epochs", 50),
        "batch_used":   params.get("batch", -1),
        "mAP50":        round(mAP50,   4),
        "mAP50_95":     round(mAP5095, 4),
        "precision":    round(prec,    4),
        "recall":       round(rec,     4),
        "f1":           round(f1,      4),
        "train_time_s": round(elapsed),
        "notes":        f"preset={preset}",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement YOLO méduses")
    parser.add_argument("--preset",  default="baseline",
                        choices=list(EXPERIMENTS.keys()),
                        help="Preset d'hyperparamètres (défaut : baseline)")
    parser.add_argument("--dataset", default=ACTIVE_DATASET,
                        choices=list(DATASET_DIRS.keys()),
                        help="Dataset à utiliser (défaut : current)")
    parser.add_argument("--model",  default=None, help="Ex : yolov8s.pt")
    parser.add_argument("--imgsz",  type=int,     default=None)
    parser.add_argument("--epochs", type=int,     default=None)
    parser.add_argument("--batch",   type=int, default=None)
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping : arrêt si pas d'amélioration pendant N epochs")
    parser.add_argument("--name",   default=None, help="Nom du run")
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("preset", "dataset") and v is not None}
    train(preset=args.preset, dataset=args.dataset, **overrides)
