"""
Comparaison entre deux fichiers de trajectoires (xlsx S1 ou CSV nouveau).

Produit un rapport Markdown dans results/comparison_<nom_ref>_vs_<nom_new>.md

Usage :
    python src/compare_trajectories.py \
        --ref  path/to/trajectoires_meduses_avec_pixels.xlsx \
        --new  results/trajectories_DJI0013_yolov8n_1280.csv \
        --fps-ref 5.0 \
        --fps-new 29.97
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR

USEFUL_COLS = ["jelly_id", "frame_idx", "x_center", "y_center",
               "width", "height", "marked", "nb_pixels"]


# ─────────────────────────────────────────────────────────────────────────────
# Chargement

def load(path: Path) -> pd.DataFrame:
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    # Garder uniquement les colonnes utiles (ignore les Unnamed:* du xlsx S1)
    cols = [c for c in USEFUL_COLS if c in df.columns]
    return df[cols].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Stats par fichier

def stats(df: pd.DataFrame, fps: float, label: str) -> dict:
    traj = df.groupby("jelly_id")
    traj_len_frames = traj["frame_idx"].count()
    traj_len_s = traj_len_frames / fps

    n_marked = df[df["marked"] == 1]["jelly_id"].nunique() if "marked" in df.columns else "N/A"

    # Couverture temporelle de chaque trajectoire (en secondes)
    traj_span_s = (traj["frame_idx"].max() - traj["frame_idx"].min()) / fps

    return {
        "label": label,
        "fps": fps,
        "total_detections": len(df),
        "n_ids": df["jelly_id"].nunique(),
        "n_marked": n_marked,
        "traj_len_frames_mean": traj_len_frames.mean(),
        "traj_len_frames_median": traj_len_frames.median(),
        "traj_len_frames_max": traj_len_frames.max(),
        "traj_len_s_mean": traj_len_s.mean(),
        "traj_len_s_median": traj_len_s.median(),
        "traj_len_s_max": traj_len_s.max(),
        "traj_span_s_mean": traj_span_s.mean(),
        "traj_span_s_median": traj_span_s.median(),
        "frame_range": (int(df["frame_idx"].min()), int(df["frame_idx"].max())),
        "traj_len_frames": traj_len_frames,
        "traj_len_s": traj_len_s,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rapport Markdown

def build_report(s_ref: dict, s_new: dict) -> str:
    lines = []

    def h(n, text): lines.append(f"{'#' * n} {text}\n")
    def row(*cols): lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    def sep(n): lines.append("| " + " | ".join(["---"] * n) + " |")

    h(1, "Comparaison des trajectoires")
    lines.append(f"- **Référence** : {s_ref['label']}  (fps={s_ref['fps']})")
    lines.append(f"- **Nouveau**   : {s_new['label']}  (fps={s_new['fps']})\n")

    h(2, "1. Statistiques globales")
    row("Métrique", s_ref["label"], s_new["label"], "Delta")
    sep(4)
    det_delta = s_new["total_detections"] - s_ref["total_detections"]
    row("Détections totales", s_ref["total_detections"], s_new["total_detections"],
        f"{det_delta:+d}")
    row("IDs uniques", s_ref["n_ids"], s_new["n_ids"],
        f"{s_new['n_ids'] - s_ref['n_ids']:+d}")
    row("IDs marquées", s_ref["n_marked"], s_new["n_marked"], "—")
    row("Frames indexées (min–max)",
        f"{s_ref['frame_range'][0]}–{s_ref['frame_range'][1]}",
        f"{s_new['frame_range'][0]}–{s_new['frame_range'][1]}", "—")
    lines.append("")

    h(2, "2. Longueur des trajectoires (en **frames**)")
    lines.append("> ⚠️ Non comparables directement : fps différents entre les deux fichiers.\n")
    row("Stat", s_ref["label"], s_new["label"])
    sep(3)
    row("Moyenne", f"{s_ref['traj_len_frames_mean']:.1f}", f"{s_new['traj_len_frames_mean']:.1f}")
    row("Médiane", f"{s_ref['traj_len_frames_median']:.1f}", f"{s_new['traj_len_frames_median']:.1f}")
    row("Max", int(s_ref["traj_len_frames_max"]), int(s_new["traj_len_frames_max"]))
    lines.append("")

    h(2, "3. Longueur des trajectoires (en **secondes** — comparable)")
    row("Stat", s_ref["label"], s_new["label"], "Delta")
    sep(4)
    row("Moyenne",
        f"{s_ref['traj_len_s_mean']:.1f}s",
        f"{s_new['traj_len_s_mean']:.1f}s",
        f"{s_new['traj_len_s_mean'] - s_ref['traj_len_s_mean']:+.1f}s")
    row("Médiane",
        f"{s_ref['traj_len_s_median']:.1f}s",
        f"{s_new['traj_len_s_median']:.1f}s",
        f"{s_new['traj_len_s_median'] - s_ref['traj_len_s_median']:+.1f}s")
    row("Max",
        f"{s_ref['traj_len_s_max']:.1f}s",
        f"{s_new['traj_len_s_max']:.1f}s",
        f"{s_new['traj_len_s_max'] - s_ref['traj_len_s_max']:+.1f}s")
    row("Couverture temporelle moyenne",
        f"{s_ref['traj_span_s_mean']:.1f}s",
        f"{s_new['traj_span_s_mean']:.1f}s",
        f"{s_new['traj_span_s_mean'] - s_ref['traj_span_s_mean']:+.1f}s")
    lines.append("")

    h(2, "4. Distribution des longueurs (en secondes)")
    lines.append("```")
    lines.append(f"{'Percentile':<12} {'Référence':>12} {'Nouveau':>12}")
    lines.append("-" * 38)
    for p in [10, 25, 50, 75, 90, 95, 100]:
        v_ref = np.percentile(s_ref["traj_len_s"], p)
        v_new = np.percentile(s_new["traj_len_s"], p)
        lines.append(f"p{p:<11} {v_ref:>11.1f}s {v_new:>11.1f}s")
    lines.append("```\n")

    h(2, "5. IDs individuels — Nouveau CSV")
    lines.append("Longueurs de toutes les trajectoires du nouveau fichier :\n")
    row("jelly_id", "frames", "durée (s)", "marquée")
    sep(4)
    df_new_raw = s_new.get("_df")
    if df_new_raw is not None:
        for jid, grp in df_new_raw.groupby("jelly_id"):
            n_fr = len(grp)
            dur = n_fr / s_new["fps"]
            mk = int(grp["marked"].max()) if "marked" in grp.columns else "?"
            row(jid, n_fr, f"{dur:.1f}s", "oui" if mk == 1 else "non")
    lines.append("")

    h(2, "6. Note méthodologique")
    lines.append(
        "- Les IDs ne sont **pas comparables** entre les deux fichiers : ByteTrack "
        "réassigne les IDs indépendamment à chaque session.\n"
        "- La référence S1 est sous-échantillonnée (~5fps effectifs), "
        "le nouveau CSV est à 29.97fps. "
        "Les durées en secondes sont donc la bonne unité de comparaison.\n"
        "- `marked` = 0 dans le nouveau CSV (non rempli automatiquement — "
        "à annoter manuellement ou via le script K-means HSV)."
    )

    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comparaison de deux fichiers de trajectoires")
    parser.add_argument("--ref",     required=True, type=Path,
                        help="Fichier de référence S1 (.xlsx ou .csv)")
    parser.add_argument("--new",     required=True, type=Path,
                        help="Nouveau fichier CSV produit par inference_video.py")
    parser.add_argument("--fps-ref", type=float, default=5.0,
                        help="FPS effectifs du fichier de référence (défaut: 5.0 = 1 frame/6 à 29.97)")
    parser.add_argument("--fps-new", type=float, default=29.97,
                        help="FPS de la vidéo source du nouveau fichier (défaut: 29.97)")
    parser.add_argument("--output",  type=Path, default=None,
                        help="Chemin du rapport Markdown (défaut: results/comparison_*.md)")
    args = parser.parse_args()

    df_ref = load(args.ref)
    df_new = load(args.new)

    s_ref = stats(df_ref, args.fps_ref, args.ref.stem)
    s_new = stats(df_new, args.fps_new, args.new.stem)
    s_new["_df"] = df_new  # pour le tableau individuel

    report = build_report(s_ref, s_new)

    if args.output is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"comparison_{args.ref.stem}_vs_{args.new.stem}.md"
    else:
        out = args.output
        out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(report, encoding="utf-8")
    print(f"[OK] Rapport écrit dans {out}")

    # Afficher les stats clés dans le terminal
    print("\n=== RÉSUMÉ ===")
    for s in (s_ref, s_new):
        print(f"\n[{s['label']}]")
        print(f"  IDs uniques       : {s['n_ids']}")
        print(f"  Détections totales: {s['total_detections']}")
        print(f"  Traj. moy. (s)    : {s['traj_len_s_mean']:.1f}s")
        print(f"  Traj. med. (s)    : {s['traj_len_s_median']:.1f}s")
        print(f"  Traj. max  (s)    : {s['traj_len_s_max']:.1f}s")


if __name__ == "__main__":
    main()
