import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CSV_COLUMNS, EXPERIMENTS_CSV, RESULTS_DIR


def setup_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def write_data_yaml(dataset_dir: Path) -> Path:
    """Écrit un data.yaml avec chemins absolus Windows-safe (slashes /)."""
    out_path = dataset_dir / "data_local.yaml"
    content = {
        "path":  dataset_dir.as_posix(),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["meduse"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True)
    return out_path


def read_results_csv(run_dir: Path) -> dict:
    """Lit la dernière ligne de results.csv généré par ultralytics."""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {}
    with open(results_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    # ultralytics met des espaces dans les noms de colonnes
    return {k.strip(): v.strip() for k, v in rows[-1].items()}


def append_experiment(row: dict) -> None:
    """Ajoute une ligne dans experiments.csv (crée le fichier si inexistant)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = EXPERIMENTS_CSV.exists()
    with open(EXPERIMENTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        row.setdefault("date", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))
        writer.writerow(row)
    print(f"[OK] Résultat loggué dans {EXPERIMENTS_CSV}")


def fmt_seconds(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s"
