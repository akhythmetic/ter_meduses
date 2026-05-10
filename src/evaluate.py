"""
Évalue un run entraîné sur le set de validation et log dans experiments.csv.

Usage :
    python src/evaluate.py --run baseline_yolov8n_640
    python src/evaluate.py --run baseline_yolov8n_640 --dataset current --no-save
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ACTIVE_DATASET, DATASET_DIRS, RUNS_DIR
from utils import append_experiment, write_data_yaml


def evaluate(run_name: str, dataset: str = ACTIVE_DATASET, save: bool = True) -> dict:
    run_dir = RUNS_DIR / "detect" / run_name
    if not run_dir.exists():
        print(f"✗  Run introuvable : {run_dir}")
        sys.exit(1)

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"✗  Poids introuvables : {best_pt}")
        sys.exit(1)

    dataset_dir = DATASET_DIRS.get(dataset)
    if dataset_dir is None or not dataset_dir.exists():
        print(f"✗  Dataset '{dataset}' introuvable : {dataset_dir}")
        sys.exit(1)

    yaml_path = write_data_yaml(dataset_dir)

    from ultralytics import YOLO
    model = YOLO(str(best_pt))

    print(f"\n── Évaluation : {run_name} ──────────────────────────────")
    print(f"   Modèle  : {best_pt}")
    print(f"   Dataset : {dataset}  ({dataset_dir})")
    print(f"──────────────────────────────────────────────────────────\n")

    metrics = model.val(data=str(yaml_path), split="val", verbose=True)

    mAP50   = float(metrics.box.map50)
    mAP5095 = float(metrics.box.map)
    prec    = float(metrics.box.mp)
    rec     = float(metrics.box.mr)
    f1      = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"\n── Métriques finales ────────────────────────────────────")
    print(f"   mAP50     = {mAP50:.4f}   (cible baseline ≈ 0.715)")
    print(f"   mAP50-95  = {mAP5095:.4f}")
    print(f"   Precision = {prec:.4f}   (cible baseline ≈ 0.65)")
    print(f"   Recall    = {rec:.4f}   (cible baseline ≈ 0.57)")
    print(f"   F1        = {f1:.4f}")
    print(f"──────────────────────────────────────────────────────────")

    # Lire les hyperparamètres depuis args.yaml du run
    model_meta = {"model": "?", "imgsz": "?", "epochs": "?", "batch_used": "?"}
    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        with open(args_yaml, encoding="utf-8") as f:
            run_args = yaml.safe_load(f)
        model_meta = {
            "model":      run_args.get("model",  "?"),
            "imgsz":      run_args.get("imgsz",  "?"),
            "epochs":     run_args.get("epochs", "?"),
            "batch_used": run_args.get("batch",  "?"),
        }

    result = {
        "date":        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "run_name":    run_name,
        "dataset":     dataset,
        "mAP50":       round(mAP50,   4),
        "mAP50_95":    round(mAP5095, 4),
        "precision":   round(prec,    4),
        "recall":      round(rec,     4),
        "f1":          round(f1,      4),
        "train_time_s": "",
        "notes":       "evaluate.py",
        **model_meta,
    }

    if save:
        append_experiment(result)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation YOLO méduses")
    parser.add_argument("--run",     required=True,
                        help="Nom du run (dossier dans runs/detect/)")
    parser.add_argument("--dataset", default=ACTIVE_DATASET,
                        choices=list(DATASET_DIRS.keys()))
    parser.add_argument("--no-save", action="store_true",
                        help="Ne pas logger dans experiments.csv")
    args = parser.parse_args()
    evaluate(args.run, dataset=args.dataset, save=not args.no_save)
