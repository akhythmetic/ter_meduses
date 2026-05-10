from pathlib import Path

# ── Racine du repo (toujours absolu, peu importe le CWD) ─────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Datasets ─────────────────────────────────────────────────────────────────
DATASET_DIRS = {
    "current":  REPO_ROOT / "data" / "current",
    "enriched": REPO_ROOT / "data" / "enriched",  # disponible dans ~5 jours
}
ACTIVE_DATASET = "current"

# ── Dossiers de sortie ────────────────────────────────────────────────────────
RUNS_DIR    = REPO_ROOT / "runs"
MODELS_DIR  = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"

# ── CSV des expériences ───────────────────────────────────────────────────────
EXPERIMENTS_CSV = RESULTS_DIR / "experiments.csv"
CSV_COLUMNS = [
    "date", "run_name", "dataset", "model", "imgsz", "epochs",
    "batch_used", "mAP50", "mAP50_95", "precision", "recall", "f1",
    "train_time_s", "notes",
]

# ── Hyperparamètres ───────────────────────────────────────────────────────────
# Baseline : reproduit les conditions du rapport S1
BASELINE = dict(
    model  = "yolov8n.pt",
    imgsz  = 640,
    epochs = 50,
    batch  = -1,    # auto (ultralytics choisit selon la VRAM)
    device = 0,     # GPU 0
    name   = "baseline_yolov8n_640",
)

# Expériences futures (ne pas lancer avant d'avoir validé le baseline)
EXPERIMENTS = {
    "baseline":   BASELINE,
    "imgsz1280":  {**BASELINE, "imgsz": 1280, "name": "exp_yolov8n_1280"},
    "v8s_640":    {**BASELINE, "model": "yolov8s.pt", "name": "exp_yolov8s_640"},
    "v8s_1280":   {**BASELINE, "model": "yolov8s.pt", "imgsz": 1280, "name": "exp_yolov8s_1280"},
    "v8m_640":    {**BASELINE, "model": "yolov8m.pt", "name": "exp_yolov8m_640"},
    "aug_medium": {
        **BASELINE,
        "name":   "exp_yolov8n_640_aug",
        "hsv_h":  0.05,
        "hsv_s":  0.7,
        "hsv_v":  0.5,
        "fliplr": 0.5,
        "flipud": 0.1,
        "scale":  0.7,
        "mosaic": 0.5,
        "mixup":  0.1,
    },
}
