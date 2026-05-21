"""Étape 3 — Préparation du split enrichi pour ré-entraînement."""

import shutil
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parent.parent
CURRENT = REPO / "data/current"
LABELS_S2 = REPO / "data/enriched/labels_S2"
OUT = REPO / "data/enriched_split"

# Session test = mAP50 OOD le plus bas (2025_08_08 = 0.002)
TEST_SESSION = "2025_08_08"
TRAIN_SESSIONS = ["2025_06_06", "2025_08_15"]


def copy_dji0013(split: str):
    """Copy DJI_0013 images and labels for a given split (train or val)."""
    src_imgs = CURRENT / "images" / split
    src_lbls = CURRENT / "labels" / split
    dst_imgs = OUT / split / "images"
    dst_lbls = OUT / split / "labels"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    n = 0
    for img in sorted(src_imgs.glob("*.jpg")):
        shutil.copy2(img, dst_imgs / img.name)
        lbl = src_lbls / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, dst_lbls / lbl.name)
        n += 1
    return n


def copy_session(session_name: str, split: str):
    """Copy a labels_S2 session into the given split with prefixed filenames."""
    session_dir = LABELS_S2 / session_name
    dst_imgs = OUT / split / "images"
    dst_lbls = OUT / split / "labels"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    n_img = n_lbl = 0
    for subdir in sorted([d for d in session_dir.iterdir() if d.is_dir()]):
        prefix = f"{session_name}_{subdir.name}"
        labels_dir = subdir / "labels"
        for img in sorted(subdir.glob("*.jpg")):
            dst_img = dst_imgs / f"{prefix}_{img.name}"
            shutil.copy2(img, dst_img)
            n_img += 1
            lbl = labels_dir / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, dst_lbls / f"{prefix}_{img.stem}.txt")
                n_lbl += 1
    return n_img, n_lbl


def verify_no_test_leakage(test_stems: set):
    """Ensure no test image stem appears in train or val."""
    leaks = []
    for split in ["train", "val"]:
        for img in (OUT / split / "images").glob("*.jpg"):
            if img.stem in test_stems:
                leaks.append((split, img.name))
    return leaks


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    print(f"Session TEST  : {TEST_SESSION}")
    print(f"Sessions TRAIN: {TRAIN_SESSIONS}")
    print()

    # --- Train ---
    n_dji_train = copy_dji0013("train")
    print(f"Train — DJI_0013 : {n_dji_train} images copiées")

    total_new_train_imgs = 0
    for sess in TRAIN_SESSIONS:
        ni, nl = copy_session(sess, "train")
        print(f"Train — {sess}  : {ni} images, {nl} labels")
        total_new_train_imgs += ni

    n_train_total = len(list((OUT / "train/images").glob("*.jpg")))
    n_train_lbls = len(list((OUT / "train/labels").glob("*.txt")))
    print(f"Train TOTAL   : {n_train_total} images, {n_train_lbls} labels")
    print()

    # --- Val (DJI_0013 inchangé) ---
    n_dji_val = copy_dji0013("val")
    n_val_total = len(list((OUT / "val/images").glob("*.jpg")))
    print(f"Val   — DJI_0013 : {n_dji_val} images copiées")
    print(f"Val   TOTAL   : {n_val_total} images")
    print()

    # --- Test ---
    ni_test, nl_test = copy_session(TEST_SESSION, "test")
    n_test_total = len(list((OUT / "test/images").glob("*.jpg")))
    print(f"Test  — {TEST_SESSION}: {ni_test} images, {nl_test} labels")
    print(f"Test  TOTAL   : {n_test_total} images")
    print()

    # --- Verify no leakage ---
    test_stems = {f.stem for f in (OUT / "test/images").glob("*.jpg")}
    leaks = verify_no_test_leakage(test_stems)
    if leaks:
        print(f"ERREUR LEAKAGE: {leaks[:5]}")
    else:
        print("Vérification anti-leakage : OK — aucune image test dans train/val")

    # --- Check expected counts ---
    print()
    print(f"Attendu train : ~{1153 + total_new_train_imgs} → obtenu {n_train_total}")
    print(f"Attendu val   : 289 → obtenu {n_val_total}")
    print(f"Attendu test  : ~140 → obtenu {n_test_total}")

    # --- YAML ---
    cfg = {
        "path": str(OUT).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": ["meduse"],
    }
    yaml_path = OUT / "data_enriched.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"\nYAML créé : {yaml_path}")
    print(f"\nContenu YAML:")
    print(open(yaml_path).read())


if __name__ == "__main__":
    main()
