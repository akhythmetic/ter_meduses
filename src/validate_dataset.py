"""
Vérifie l'intégrité du dataset avant entraînement.

Usage :
    python src/validate_dataset.py
    python src/validate_dataset.py --dataset enriched
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ACTIVE_DATASET, DATASET_DIRS

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _img_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def _label_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix == ".txt")


# ─────────────────────────────────────────────────────────────────────────────

def check_structure(dataset_dir: Path) -> bool:
    required = [
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
        dataset_dir / "data.yaml",
    ]
    ok = True
    for p in required:
        exists = p.exists()
        mark = "✓" if exists else "✗"
        print(f"  {mark}  {p.relative_to(dataset_dir)}")
        if not exists:
            ok = False
    return ok


def check_data_yaml(dataset_dir: Path) -> bool:
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ok = True
    nc    = cfg.get("nc", 0)
    names = cfg.get("names", [])

    if nc == 1 and names == ["meduse"]:
        print(f"  ✓  nc={nc}, names={names}")
    else:
        print(f"  ✗  nc={nc}, names={names}  (attendu: nc=1, names=['meduse'])")
        ok = False

    train_path = str(cfg.get("train", ""))
    val_path   = str(cfg.get("val",   ""))
    if "/content/" in train_path or "/content/" in val_path:
        print(f"  ⚠  Chemins Colab détectés → train.py génère data_local.yaml automatiquement")
    elif dataset_dir.as_posix() in train_path:
        print(f"  ✓  Chemins absolus présents")
    else:
        print(f"  ⚠  Chemins relatifs/inconnus → train.py génère data_local.yaml automatiquement")

    return ok


def count_and_match(dataset_dir: Path, split: str) -> dict:
    imgs_dir   = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    imgs   = {p.stem: p for p in _img_files(imgs_dir)}
    labels = {p.stem: p for p in _label_files(labels_dir)}

    only_img   = sorted(set(imgs)   - set(labels))
    only_label = sorted(set(labels) - set(imgs))

    status = "✓" if not only_img and not only_label else "⚠"
    msg = f"  {status}  {split:6} : {len(imgs):4d} images  /  {len(labels):4d} labels"
    if only_img:
        msg += f"  [{len(only_img)} images sans label]"
    if only_label:
        msg += f"  [{len(only_label)} labels sans image]"
    print(msg)

    if only_img:
        for stem in only_img[:3]:
            print(f"       image sans label : {stem}")
        if len(only_img) > 3:
            print(f"       ... et {len(only_img)-3} autres")
    if only_label:
        for stem in only_label[:3]:
            print(f"       label sans image : {stem}")
        if len(only_label) > 3:
            print(f"       ... et {len(only_label)-3} autres")

    return {"n_images": len(imgs), "n_labels": len(labels),
            "orphan_images": only_img, "orphan_labels": only_label}


def validate_labels(dataset_dir: Path, split: str) -> dict:
    """Vérifie le format YOLO et calcule des stats sur les boîtes."""
    labels_dir = dataset_dir / "labels" / split
    label_files = _label_files(labels_dir)

    errors     = []
    empty      = 0
    total_boxes = 0
    areas      = []   # w*h (fraction de l'image)
    widths     = []
    heights    = []

    for lf in label_files:
        content = lf.read_text(encoding="utf-8").strip()
        if not content:
            empty += 1
            continue
        for i, line in enumerate(content.splitlines(), 1):
            parts = line.strip().split()
            if len(parts) != 5:
                errors.append(f"{lf.name}:{i} — {len(parts)} champs (attendu 5)")
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(f"{lf.name}:{i} — valeurs non numériques")
                continue
            if cls != 0:
                errors.append(f"{lf.name}:{i} — classe {cls} (attendu 0)")
            if not (0 < cx <= 1 and 0 < cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                errors.append(
                    f"{lf.name}:{i} — coords hors (0,1] : cx={cx:.4f} cy={cy:.4f} "
                    f"w={w:.4f} h={h:.4f}"
                )
            total_boxes += 1
            areas.append(w * h)
            widths.append(w)
            heights.append(h)

    status = "✓" if not errors else "⚠"
    print(f"  {status}  {split:6} : {len(label_files):4d} labels vérifiés  |  "
          f"vides (0 méduses) : {empty}  |  erreurs format : {len(errors)}")

    if errors:
        for e in errors[:5]:
            print(f"       ✗ {e}")
        if len(errors) > 5:
            print(f"       ... et {len(errors)-5} autres erreurs")

    if areas:
        a  = np.array(areas)
        ws = np.array(widths)
        hs = np.array(heights)
        print(f"    ── Boîtes {split} ({total_boxes} total) ──────────────────────")
        print(f"       area%  : min={a.min()*100:.3f}%  p25={np.percentile(a,25)*100:.3f}%  "
              f"méd={np.median(a)*100:.3f}%  p75={np.percentile(a,75)*100:.3f}%  "
              f"max={a.max()*100:.3f}%")
        print(f"       w (rel): min={ws.min():.4f}  méd={np.median(ws):.4f}  max={ws.max():.4f}")
        print(f"       h (rel): min={hs.min():.4f}  méd={np.median(hs):.4f}  max={hs.max():.4f}")
        # Estimation pixels à imgsz=640 (largeur et hauteur images supposées ~640)
        print(f"       w_px @640 : min={ws.min()*640:.0f}px  méd={np.median(ws)*640:.0f}px  "
              f"max={ws.max()*640:.0f}px")
        print(f"       h_px @640 : min={hs.min()*640:.0f}px  méd={np.median(hs)*640:.0f}px  "
              f"max={hs.max()*640:.0f}px")

    return {"n_boxes": total_boxes, "n_empty": empty, "n_errors": len(errors)}


# ─────────────────────────────────────────────────────────────────────────────

def run(dataset_name: str = ACTIVE_DATASET) -> bool:
    dataset_dir = DATASET_DIRS.get(dataset_name)
    if dataset_dir is None:
        print(f"Dataset inconnu : {dataset_name}  (choix : {list(DATASET_DIRS)})")
        return False
    if not dataset_dir.exists():
        print(f"✗  Dossier introuvable : {dataset_dir}")
        return False

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Validation du dataset : {dataset_name}")
    print(f"  {dataset_dir}")
    print(sep)

    print("\n[1] Structure des dossiers")
    if not check_structure(dataset_dir):
        print("  → Structure invalide, arrêt.")
        return False

    print("\n[2] data.yaml")
    check_data_yaml(dataset_dir)

    print("\n[3] Comptage et matching images ↔ labels")
    for split in ("train", "val"):
        count_and_match(dataset_dir, split)

    print("\n[4] Format YOLO et stats des boîtes")
    for split in ("train", "val"):
        validate_labels(dataset_dir, split)

    print(f"\n{sep}")
    print("  Validation terminée.")
    print(sep)
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validation du dataset YOLO méduses")
    parser.add_argument("--dataset", default=ACTIVE_DATASET,
                        choices=list(DATASET_DIRS.keys()))
    args = parser.parse_args()

    ok = run(args.dataset)
    sys.exit(0 if ok else 1)
