"""
Inférence + tracking ByteTrack sur une vidéo entière.

Produit :
  - CSV de trajectoires (jelly_id, frame_idx, x_center, y_center,
                          width, height, marked, nb_pixels)
  - Vidéo annotée (optionnel)

Usage :
    python src/inference_video.py \
        --video  path/to/DJI_0013.mp4 \
        --model  runs/detect/yolov8n_1280_e150/weights/best.pt \
        --output results/trajectories_yolov8n_1280_e150_DJI0013.csv \
        --conf   0.25 \
        --annotated-video results/DJI0013_annotated.mp4
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import fmt_seconds


# ─────────────────────────────────────────────────────────────────────────────

def run(
    video_path: Path,
    model_path: Path,
    output_csv: Path,
    conf: float = 0.25,
    annotated_video_path: Path | None = None,
    imgsz: int = 1280,
    output_scale: float = 0.5,
    vid_stride: int = 1,
) -> None:
    from ultralytics import YOLO

    # ── Vérifications ────────────────────────────────────────────────────────
    if not video_path.exists():
        print(f"[ERREUR] Vidéo introuvable : {video_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"[ERREUR] Modèle introuvable : {model_path}")
        sys.exit(1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── Infos vidéo ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    effective_fps = fps / vid_stride
    effective_frames = (total + vid_stride - 1) // vid_stride

    print(f"[INFO] Vidéo   : {video_path.name}")
    print(f"[INFO] Résol.  : {width}x{height}  {fps:.2f} fps  {total} frames")
    print(f"[INFO] Stride  : {vid_stride}  ({effective_frames} frames traitées, {effective_fps:.2f} fps effectifs)")
    print(f"[INFO] Modèle  : {model_path}")
    print(f"[INFO] Sortie  : {output_csv}")
    if annotated_video_path:
        print(f"[INFO] Vidéo annotée : {annotated_video_path}")
    print()

    # ── Initialisation du writer vidéo (optionnel) ───────────────────────────
    writer = None
    out_w, out_h = width, height
    if annotated_video_path:
        annotated_video_path.parent.mkdir(parents=True, exist_ok=True)
        if output_scale != 1.0:
            out_w = int(width * output_scale)
            out_h = int(height * output_scale)
        out_fps = fps / vid_stride  # préserver la durée réelle (ex: 5fps pour stride=6)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(annotated_video_path), fourcc, out_fps, (out_w, out_h)
        )
        print(f"[INFO] Vidéo sortie : {out_w}x{out_h} | {out_fps:.2f}fps (scale={output_scale}, stride={vid_stride})")

    # ── Tracking ByteTrack ───────────────────────────────────────────────────
    model = YOLO(str(model_path))

    CSV_COLS = ["jelly_id", "frame_idx", "x_center", "y_center",
                "width", "height", "marked", "nb_pixels"]

    t0 = time.time()
    rows_written = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer_csv.writeheader()

        results_gen = model.track(
            source=str(video_path),
            conf=conf,
            imgsz=imgsz,
            tracker="bytetrack.yaml",
            stream=True,        # génère frame par frame sans tout charger en RAM
            persist=True,       # maintient les IDs entre frames
            verbose=False,
            vid_stride=vid_stride,
        )

        for frame_idx, result in enumerate(results_gen):
            # ── Écriture CSV ────────────────────────────────────────────────
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes
                ids   = boxes.id  # None si pas de tracking (ne devrait pas arriver)

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    bw   = float(x2 - x1)
                    bh   = float(y2 - y1)
                    cx   = float(x1 + bw / 2)
                    cy   = float(y1 + bh / 2)
                    jid  = int(ids[i].item()) if ids is not None else -1

                    writer_csv.writerow({
                        "jelly_id":  jid,
                        "frame_idx": frame_idx,
                        "x_center":  round(cx, 2),
                        "y_center":  round(cy, 2),
                        "width":     round(bw, 2),
                        "height":    round(bh, 2),
                        "marked":    0,             # rempli manuellement ensuite
                        "nb_pixels": round(bw * bh, 2),
                    })
                    rows_written += 1

            # ── Vidéo annotée ───────────────────────────────────────────────
            if writer is not None:
                frame = result.plot()
                if (out_w, out_h) != (width, height):
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                writer.write(frame)

            # ── Progression ─────────────────────────────────────────────────
            if frame_idx % 200 == 0 and frame_idx > 0:
                elapsed = time.time() - t0
                eta     = elapsed / frame_idx * (effective_frames - frame_idx) if effective_frames else 0
                print(f"  frame {frame_idx:5d}/{effective_frames}  |  "
                      f"{frame_idx/elapsed:.1f} fps  |  ETA {fmt_seconds(eta)}  |  "
                      f"{rows_written} detections")

    if writer is not None:
        writer.release()

    elapsed = time.time() - t0
    print(f"\n[OK] Terminé en {fmt_seconds(elapsed)}")
    print(f"[OK] {rows_written} détections écrites dans {output_csv}")
    if annotated_video_path:
        print(f"[OK] Vidéo annotée : {annotated_video_path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inférence + tracking ByteTrack sur vidéo"
    )
    parser.add_argument(
        "--video", required=True, type=Path,
        help="Chemin vers la vidéo source (.mp4)"
    )
    parser.add_argument(
        "--model",
        default=str(
            Path(__file__).resolve().parent.parent
            / "runs" / "detect" / "yolov8n_1280_e150" / "weights" / "best.pt"
        ),
        type=Path,
        help="Chemin vers best.pt (défaut : yolov8n_1280_e150)"
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parent.parent
            / "results" / "trajectories_yolov8n_1280_e150_DJI0013.csv"
        ),
        type=Path,
        help="Chemin du CSV de sortie"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Seuil de confiance (défaut : 0.25)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280,
        help="Taille d'entrée pour l'inférence (défaut : 1280)"
    )
    parser.add_argument(
        "--annotated-video", type=Path, default=None,
        dest="annotated_video",
        help="Si spécifié, sauvegarde une vidéo annotée à ce chemin"
    )
    parser.add_argument(
        "--output-scale", type=float, default=0.5,
        dest="output_scale",
        help="Facteur de redimensionnement de la vidéo annotée (défaut: 0.5 = 1920x1080 pour 4K source)"
    )
    parser.add_argument(
        "--vid-stride", type=int, default=1,
        dest="vid_stride",
        help="Traiter 1 frame sur N (défaut: 1 = toutes les frames, 6 = ~5fps effectifs)"
    )
    args = parser.parse_args()

    run(
        video_path           = args.video,
        model_path           = args.model,
        output_csv           = args.output,
        conf                 = args.conf,
        imgsz                = args.imgsz,
        annotated_video_path = args.annotated_video,
        output_scale         = args.output_scale,
        vid_stride           = args.vid_stride,
    )
