"""
Extrait les clips video des fenetres de validation (Step 2).
- Crop 400x400 centre sur x_center_moyen/y_center_moyen, borne aux dims video
- Rescale en 800x800
- Overlay texte "W01 - B3 - 40.0s-49.8s" (bas gauche, blanc avec ombre)
- Sauvegarde results/clips/WXX_balisee.mp4 a 29.97 fps

Usage:
    python src/extract_window_clips.py
    python src/extract_window_clips.py --video "TER analyse videos meduses/2025_06_30/DJI_0013.MP4"
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = REPO / "TER analyse videos meduses" / "2025_06_30" / "DJI_0013.MP4"
CSV_IN    = REPO / "results" / "windows_selected.csv"
CLIPS_DIR = REPO / "results" / "clips"

CROP_SIZE    = 400
OUT_SIZE     = 800
FPS_OUT      = 29.97

FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.8
FONT_COLOR   = (255, 255, 255)
FONT_THICK   = 2
SHADOW_COLOR = (0, 0, 0)


def clamp_crop(cx: float, cy: float, w: int, h: int) -> tuple[int, int, int, int]:
    half = CROP_SIZE // 2
    x1 = int(cx) - half
    x2 = x1 + CROP_SIZE
    if x1 < 0:
        x1, x2 = 0, CROP_SIZE
    if x2 > w:
        x2, x1 = w, w - CROP_SIZE
    y1 = int(cy) - half
    y2 = y1 + CROP_SIZE
    if y1 < 0:
        y1, y2 = 0, CROP_SIZE
    if y2 > h:
        y2, y1 = h, h - CROP_SIZE
    return x1, y1, x2, y2


def extract_clip(
    cap: cv2.VideoCapture,
    frame_start: int,
    frame_end: int,
    cx: float,
    cy: float,
    orig_w: int,
    orig_h: int,
    label: str,
    out_path: Path,
) -> None:
    x1, y1, x2, y2 = clamp_crop(cx, cy, orig_w, orig_h)
    n_frames = frame_end - frame_start + 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS_OUT, (OUT_SIZE, OUT_SIZE))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    for _ in tqdm(range(n_frames), desc=label, leave=False, unit="fr"):
        ret, frame = cap.read()
        if not ret:
            break
        crop    = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)

        text_y = OUT_SIZE - 12
        cv2.putText(resized, label, (11, text_y + 1), FONT, FONT_SCALE, SHADOW_COLOR, FONT_THICK + 1, cv2.LINE_AA)
        cv2.putText(resized, label, (10, text_y),     FONT, FONT_SCALE, FONT_COLOR,   FONT_THICK,     cv2.LINE_AA)

        writer.write(resized)

    writer.release()


def main(video_path: Path) -> None:
    if not video_path.exists():
        print(f"ERREUR : video introuvable : {video_path}")
        return

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    df  = pd.read_csv(CSV_IN)
    cap = cv2.VideoCapture(str(video_path))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video : {orig_w}x{orig_h}")
    print(f"Extraction de {len(df)} clips vers {CLIPS_DIR}/")

    for _, row in df.iterrows():
        wid     = row["window_id"]
        bp      = row["balisee_physique"]
        fs      = int(row["frame_start"])
        fe      = int(row["frame_end"])
        cx      = float(row["x_center_moyen"])
        cy      = float(row["y_center_moyen"])
        t0      = float(row["time_start_s"])
        t1      = float(row["time_end_s"])

        safe_bp  = bp.replace("/", "_").replace(" ", "_")
        out_path = CLIPS_DIR / f"{wid}_{safe_bp}.mp4"

        label = f"{wid} - {bp} - {t0:.1f}s-{t1:.1f}s"
        print(f"  {label}")
        extract_clip(cap, fs, fe, cx, cy, orig_w, orig_h, label, out_path)
        print(f"    -> {out_path.name}")

    cap.release()
    print(f"\n[OK] {len(df)} clips extraits dans {CLIPS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO, type=Path)
    args = parser.parse_args()
    main(args.video)
