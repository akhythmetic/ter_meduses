"""Étape 2 — Évaluation OOD du modèle DJI_0013 sur les nouvelles sessions."""

import shutil
import csv
import json
from pathlib import Path
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO / "runs/detect/yolov8n_1280_e150/weights/best.pt"
LABELS_S2 = REPO / "data/enriched/labels_S2"
TMP_DIR = REPO / "data/tmp_ood"
RESULTS_DIR = REPO / "results"
FIGURES_DIR = REPO / "figures"
LOGS_DIR = REPO / "logs"

for d in [TMP_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def build_flat_dataset(session_name: str, session_dirs: list[Path], out_dir: Path):
    """Flatten multi-subdir session into images/ + labels/ with prefixed filenames."""
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_img = n_lbl = 0
    for subdir in session_dirs:
        prefix = subdir.name
        labels_dir = subdir / "labels"
        for img in sorted(subdir.glob("*.jpg")):
            dst = img_out / f"{prefix}_{img.name}"
            shutil.copy2(img, dst)
            n_img += 1
            lbl = labels_dir / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, lbl_out / f"{prefix}_{lbl.name}")
                n_lbl += 1
    return n_img, n_lbl


def make_yaml(out_dir: Path, dataset_path: Path) -> Path:
    yaml_path = out_dir / "data.yaml"
    cfg = {
        "path": str(dataset_path),
        "train": "images",
        "val": "images",
        "nc": 1,
        "names": ["meduse"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return yaml_path


def run_val(model: YOLO, yaml_path: Path, session_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluation: {session_name}")
    metrics = model.val(
        data=str(yaml_path),
        imgsz=1280,
        conf=0.25,
        split="val",
        verbose=False,
        plots=False,
        save_json=False,
        project=str(REPO / "runs/ood_eval"),
        name=session_name,
        exist_ok=True,
    )
    return {
        "session": session_name,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }


def main():
    model = YOLO(str(MODEL_PATH))
    sessions = sorted([d for d in LABELS_S2.iterdir() if d.is_dir()])

    all_results = []

    # --- Per-session evaluation ---
    all_imgs_dir = TMP_DIR / "all_sessions" / "images"
    all_lbls_dir = TMP_DIR / "all_sessions" / "labels"
    all_imgs_dir.mkdir(parents=True, exist_ok=True)
    all_lbls_dir.mkdir(parents=True, exist_ok=True)

    for session in sessions:
        subdirs = sorted([d for d in session.iterdir() if d.is_dir()])
        out_dir = TMP_DIR / session.name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        n_img, n_lbl = build_flat_dataset(session.name, subdirs, out_dir)
        print(f"[{session.name}] {n_img} images, {n_lbl} labels → {out_dir}")

        # Copy to combined dir too
        for f in (out_dir / "images").glob("*.jpg"):
            shutil.copy2(f, all_imgs_dir / f.name)
        for f in (out_dir / "labels").glob("*.txt"):
            shutil.copy2(f, all_lbls_dir / f.name)

        yaml_path = make_yaml(out_dir, out_dir)
        res = run_val(model, yaml_path, session.name)
        res["n_images"] = n_img
        res["n_labels"] = n_lbl
        all_results.append(res)

    # --- Combined evaluation ---
    all_dir = TMP_DIR / "all_sessions"
    yaml_all = make_yaml(all_dir, all_dir)
    res_all = run_val(model, yaml_all, "all_sessions_OOD")
    n_all_img = len(list(all_imgs_dir.glob("*.jpg")))
    n_all_lbl = len(list(all_lbls_dir.glob("*.txt")))
    res_all["n_images"] = n_all_img
    res_all["n_labels"] = n_all_lbl
    all_results.append(res_all)

    # --- Save CSV ---
    csv_path = RESULTS_DIR / "ood_evaluation_before.csv"
    fieldnames = ["session", "mAP50", "mAP50_95", "precision", "recall", "n_images", "n_labels"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV sauvegardé: {csv_path}")

    # --- Print comparison table ---
    DJI0013 = {"mAP50": 0.96, "precision": 0.89, "recall": 0.90}
    print("\n" + "="*72)
    print(f"{'Session':<22} {'mAP50':>7} {'mAP50-95':>9} {'Precision':>10} {'Recall':>8} {'Imgs':>6}")
    print("-"*72)
    print(f"{'DJI_0013 (référence)':<22} {'0.960':>7} {'—':>9} {'0.890':>10} {'0.900':>8} {'1153':>6}")
    for r in all_results:
        print(f"{r['session']:<22} {r['mAP50']:>7.3f} {r['mAP50_95']:>9.3f} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['n_images']:>6}")
    print("="*72)

    # --- Figure ---
    sessions_names = [r["session"] for r in all_results]
    map50_vals = [r["mAP50"] for r in all_results]
    prec_vals = [r["precision"] for r in all_results]
    rec_vals = [r["recall"] for r in all_results]

    # Add DJI_0013 reference at front
    all_labels = ["DJI_0013\n(référence)"] + sessions_names
    all_map50 = [0.96] + map50_vals
    all_prec = [0.89] + prec_vals
    all_rec = [0.90] + rec_vals

    x = np.arange(len(all_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, all_map50, width, label="mAP50", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x, all_prec, width, label="Precision", color="#4CAF50", alpha=0.85)
    bars3 = ax.bar(x + width, all_rec, width, label="Recall", color="#FF9800", alpha=0.85)
    ax.axhline(y=0.70, color="red", linestyle="--", alpha=0.6, label="seuil 0.70")
    ax.set_ylabel("Score")
    ax.set_title("Évaluation OOD — Modèle DJI_0013 sur nouvelles sessions\n(avant ré-entraînement)")
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "ood_evaluation_before.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figure sauvegardée: {fig_path}")

    # --- Log ---
    with open(LOGS_DIR / "ood_evaluation.log", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    main()
