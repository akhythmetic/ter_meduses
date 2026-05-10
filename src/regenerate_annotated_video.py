"""
Régénère une vidéo annotée à partir d'une vidéo source + CSV de trajectoires.
N'exécute pas YOLO — lit les détections du CSV et les dessine sur les frames.

Usage :
    python src/regenerate_annotated_video.py \
        --video  "TER analyse videos meduses/2025_06_30/DJI_0013.MP4" \
        --csv    results/trajectories_DJI0013_yolov8n_1280_stride6.csv \
        --output results/DJI0013_annotated_stride6.mp4 \
        --scale  0.5 \
        --stride 6
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def id_color(jelly_id: int) -> tuple:
    """Couleur BGR reproductible par ID."""
    rng = np.random.default_rng(abs(jelly_id) + 1)
    h = int(rng.integers(0, 180))
    hsv = np.uint8([[[h, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def fmt_seconds(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s"


def run(video_path: Path, csv_path: Path, output_path: Path,
        scale: float = 0.5, stride: int = 1) -> None:

    if not video_path.exists():
        raise FileNotFoundError(f"Vidéo introuvable : {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    # ── Charger le CSV et indexer par frame_idx ───────────────────────────────
    df = pd.read_csv(csv_path)
    # Grouper les détections par frame pour un accès O(1)
    detections = {fi: grp for fi, grp in df.groupby("frame_idx")}
    max_frame_idx = int(df["frame_idx"].max())

    # ── Infos vidéo source ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    src_fps    = cap.get(cv2.CAP_PROP_FPS)
    src_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = int(src_width  * scale)
    out_h = int(src_height * scale)
    out_fps = src_fps / stride

    n_processed = (src_total + stride - 1) // stride

    print(f"[INFO] Source  : {video_path.name}  {src_width}x{src_height}  {src_fps:.2f}fps  {src_total} frames")
    print(f"[INFO] Stride  : {stride}  -> {n_processed} frames a traiter  ({out_fps:.2f}fps effectifs)")
    print(f"[INFO] CSV     : {len(df)} détections  frame_idx 0–{max_frame_idx}")
    print(f"[INFO] Sortie  : {output_path}  {out_w}x{out_h}  {out_fps:.2f}fps")
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (out_w, out_h))

    t0 = time.time()
    frame_idx   = 0   # compteur de frames traitées (correspond au frame_idx du CSV)
    video_frame = 0   # compteur de frames lues dans la vidéo source

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if video_frame % stride == 0:
            # Redimensionner
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            # Dessiner les détections de cette frame
            if frame_idx in detections:
                for _, det in detections[frame_idx].iterrows():
                    jid = int(det["jelly_id"])
                    # Coordonnées en pixels source → scale
                    cx = float(det["x_center"]) * scale
                    cy = float(det["y_center"]) * scale
                    bw = float(det["width"])    * scale
                    bh = float(det["height"])   * scale
                    x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                    x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

                    color = id_color(jid)
                    marked = int(det.get("marked", 0))
                    thickness = 3 if marked else 2

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                    label = f"ID:{jid}" + (" [M]" if marked else "")
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    lx = max(x1, 0)
                    ly = max(y1 - 4, th + 2)
                    cv2.rectangle(frame, (lx, ly - th - 2), (lx + tw, ly + 2), color, -1)
                    cv2.putText(frame, label, (lx, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 200 == 0:
                elapsed = time.time() - t0
                eta = elapsed / frame_idx * (n_processed - frame_idx)
                print(f"  frame {frame_idx:5d}/{n_processed}  |  {frame_idx/elapsed:.1f} fps  |  ETA {fmt_seconds(eta)}")

        video_frame += 1

    cap.release()
    writer.release()

    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / 1e6

    # Vérification durée réelle
    check = cv2.VideoCapture(str(output_path))
    out_nf  = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps_check = check.get(cv2.CAP_PROP_FPS)
    check.release()
    out_dur = out_nf / out_fps_check if out_fps_check > 0 else 0

    print(f"\n[OK] Terminé en {fmt_seconds(elapsed)}")
    print(f"[OK] {frame_idx} frames écrites | {out_nf} frames lues en vérification")
    print(f"[OK] Durée : {out_dur:.1f}s ({out_dur/60:.2f}min) | Taille : {size_mb:.1f} MB")
    print(f"[OK] Sortie : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Régénère vidéo annotée depuis CSV")
    parser.add_argument("--video",  required=True, type=Path)
    parser.add_argument("--csv",    required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale",  type=float, default=0.5,
                        help="Facteur de résolution (défaut: 0.5 → 1920x1080 pour 4K)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride utilisé lors de l'inférence (défaut: 1)")
    args = parser.parse_args()
    run(args.video, args.csv, args.output, args.scale, args.stride)
