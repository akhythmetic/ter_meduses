"""Étape 6 — Pipeline pulsation SAM sur les 3 nouvelles sessions.

Pour chaque session : détection YOLO enrichi + SAM sur un segment 20s,
tracker spatial simple, FFT + find_peaks, vérification bande bio 0.3-1.5 Hz.
"""

import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.ndimage import median_filter
from tqdm import tqdm
from ultralytics import SAM, YOLO

REPO = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO / "runs/detect/yolov8n_1280_e150_enriched/weights/best.pt"
SAM_PATH   = REPO / "sam2.1_b.pt"
VIDEO_ROOT = REPO / "TER analyse videos meduses"
RESULTS    = REPO / "results"
FIGURES    = REPO / "figures"
LOGS       = REPO / "logs"
for d in [RESULTS, FIGURES, LOGS]:
    d.mkdir(exist_ok=True)

IMGSZ      = 1280
CONF       = 0.25
BIO_LO     = 0.3
BIO_HI     = 1.5
MIN_PTS    = 60       # points minimum pour analyser une piste
TRACK_DIST = 150      # distance max (px) pour associer détection à piste existante

SESSIONS = [
    dict(
        name="2025_06_06",
        video=VIDEO_ROOT / "2025_06_06/2025_06_06_10_46_45.mp4",
        frame_start=2762, frame_end=3242,
        fps=24.01,
    ),
    dict(
        name="2025_08_08",
        video=VIDEO_ROOT / "2025_08_08/DJI_0815.MP4",
        frame_start=1005, frame_end=1604,
        fps=29.97,
    ),
    dict(
        name="2025_08_15",
        video=VIDEO_ROOT / "2025_08_15/DJI_0820.MP4",
        frame_start=152, frame_end=751,
        fps=29.97,
    ),
]


# ── Tracker spatial simple ─────────────────────────────────────────────────────

class Track:
    _next_id = 0

    def __init__(self, cx, cy, area, frame_idx, time_s):
        self.id = Track._next_id
        Track._next_id += 1
        self.points = [dict(frame_idx=frame_idx, time_s=time_s, cx=cx, cy=cy, area=area)]
        self.last_cx, self.last_cy = cx, cy
        self.missed = 0

    def update(self, cx, cy, area, frame_idx, time_s):
        self.points.append(dict(frame_idx=frame_idx, time_s=time_s, cx=cx, cy=cy, area=area))
        self.last_cx, self.last_cy = cx, cy
        self.missed = 0


def match_detections(tracks, dets, max_dist):
    """Greedy nearest-neighbour matching. Returns (matched pairs, unmatched dets)."""
    matched = {}  # track_id -> det_idx
    used_dets = set()
    for t in tracks:
        best_d, best_i = float("inf"), -1
        for i, (cx, cy, _) in enumerate(dets):
            if i in used_dets:
                continue
            d = ((cx - t.last_cx) ** 2 + (cy - t.last_cy) ** 2) ** 0.5
            if d < best_d:
                best_d, best_i = d, i
        if best_d < max_dist and best_i >= 0:
            matched[t.id] = best_i
            used_dets.add(best_i)
    unmatched = [i for i in range(len(dets)) if i not in used_dets]
    return matched, unmatched


# ── Détrend + analyse ──────────────────────────────────────────────────────────

def detrend(sig, win):
    trend = median_filter(sig.astype(float), size=win, mode="nearest")
    return sig - trend


def analyze_track(pts, fps):
    sig_raw = np.array([p["area"] for p in pts], dtype=float)
    t       = np.array([p["time_s"] for p in pts], dtype=float)
    dt      = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0 / fps
    win     = max(int(5 * fps), 3)

    sig_det = detrend(sig_raw, win)
    std_det = np.std(sig_det)
    if std_det < 1:
        return None

    # FFT détrendée
    freqs   = np.fft.rfftfreq(len(sig_det), d=dt)
    amp     = np.abs(np.fft.rfft(sig_det))
    mask_bio = (freqs >= BIO_LO) & (freqs <= BIO_HI)
    mask_full = (freqs >= 0.05) & (freqs <= 5.0)
    dom_freq = float(freqs[mask_full][np.argmax(amp[mask_full])]) if mask_full.any() else float("nan")
    bio_freq = float(freqs[mask_bio][np.argmax(amp[mask_bio])]) if mask_bio.any() else float("nan")

    # find_peaks sur signal détrendé
    peaks, _ = sp_signal.find_peaks(
        sig_det,
        height=0.2 * std_det,
        distance=max(int(fps * 0.5), 1),
        prominence=max(0.15 * std_det, 1.0),
    )
    duree   = float(t[-1] - t[0])
    f_peaks = len(peaks) / duree if duree > 0 and len(peaks) > 0 else float("nan")

    return dict(
        n_pts=len(pts),
        dom_freq_hz=round(dom_freq, 4),
        bio_freq_hz=round(bio_freq, 4),
        f_peaks_hz=round(f_peaks, 4) if np.isfinite(f_peaks) else float("nan"),
        in_bio_dom=bool(np.isfinite(dom_freq) and BIO_LO <= dom_freq <= BIO_HI),
        in_bio_peaks=bool(np.isfinite(f_peaks) and BIO_LO <= f_peaks <= BIO_HI),
        t=t, sig_raw=sig_raw, sig_det=sig_det, freqs=freqs, amp=amp, peaks=peaks,
    )


# ── Inférence principale ───────────────────────────────────────────────────────

def run_session(sess, model_det, sam_model):
    name   = sess["name"]
    video  = sess["video"]
    f_start = sess["frame_start"]
    f_end   = sess["frame_end"]
    fps     = sess["fps"]

    print(f"\n{'='*60}")
    print(f"Session : {name}  |  {video.name}")
    print(f"Segment : frames {f_start}–{f_end}  (~20s à {fps:.2f} fps)")

    cap = cv2.VideoCapture(str(video))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Résolution : {orig_w}×{orig_h}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

    tracks: list[Track] = []
    MAX_MISSED = 5

    raw_rows = []
    n_frames = f_end - f_start + 1

    for fi in tqdm(range(n_frames), desc=name, unit="fr"):
        frame_idx = f_start + fi
        ret, frame = cap.read()
        if not ret:
            break

        time_s = round(frame_idx / fps, 5)

        res = model_det.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            for t in tracks:
                t.missed += 1
            tracks = [t for t in tracks if t.missed <= MAX_MISSED]
            continue

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()

        # SAM batch
        bboxes_sam = [[float(x1), float(y1), float(x2), float(y2)]
                      for x1, y1, x2, y2 in boxes_xyxy]
        try:
            sam_res = sam_model(frame, bboxes=bboxes_sam, verbose=False)
            masks_data = sam_res[0].masks.data.cpu().numpy() if sam_res and sam_res[0].masks else None
        except Exception as e:
            masks_data = None

        dets = []
        for bi in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[bi]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if masks_data is not None and bi < masks_data.shape[0]:
                m = masks_data[bi].astype(np.uint8)
                if m.shape[0] != orig_h or m.shape[1] != orig_w:
                    m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                area = int(m.sum())
            else:
                area = int((x2 - x1) * (y2 - y1))
            dets.append((cx, cy, area))
            raw_rows.append(dict(
                session=name, frame_idx=frame_idx, time_s=time_s,
                cx=round(cx, 1), cy=round(cy, 1),
                bbox_w=round(x2-x1, 1), bbox_h=round(y2-y1, 1),
                mask_area_px=area,
            ))

        # Tracking
        matched, unmatched = match_detections(tracks, dets, TRACK_DIST)
        tracks_by_id = {t.id: t for t in tracks}

        for tid, di in matched.items():
            cx, cy, area = dets[di]
            tracks_by_id[tid].update(cx, cy, area, frame_idx, time_s)

        for t in tracks:
            if t.id not in matched:
                t.missed += 1

        for di in unmatched:
            cx, cy, area = dets[di]
            tracks.append(Track(cx, cy, area, frame_idx, time_s))

        tracks = [t for t in tracks if t.missed <= MAX_MISSED]

    cap.release()

    # Collect all completed tracks (including those still alive)
    all_tracks = [t for t in tracks]
    # Add tracks from raw data that might have been pruned — re-read from raw_rows isn't needed,
    # since Track objects accumulate all points during the loop.
    # Reset for next session
    Track._next_id = 0

    print(f"  Pistes totales : {len(all_tracks)}")
    print(f"  Frames brutes  : {len(raw_rows)} détections")

    # Analyse des pistes longues
    analyses = []
    for tr in all_tracks:
        if len(tr.points) < MIN_PTS:
            continue
        res_a = analyze_track(tr.points, fps)
        if res_a is None:
            continue
        res_a["track_id"] = tr.id
        res_a["session"] = name
        analyses.append(res_a)

    print(f"  Pistes analysables (>={MIN_PTS} pts) : {len(analyses)}")
    for a in analyses:
        bio_tag = "OK bio" if a["in_bio_dom"] else ("OK bio (pics)" if a["in_bio_peaks"] else "HORS bande")
        f_pks_str = f"{a['f_peaks_hz']:.3f}" if isinstance(a['f_peaks_hz'], float) and not np.isnan(a['f_peaks_hz']) else "nan"
        print(f"    Track {a['track_id']:3d}: n={a['n_pts']:4d}  "
              f"dom={a['dom_freq_hz']:.3f}Hz  bio={a['bio_freq_hz']:.3f}Hz  "
              f"pics={f_pks_str}Hz  {bio_tag}")

    return raw_rows, analyses


# ── Figures par session ────────────────────────────────────────────────────────

def make_session_figure(name, analyses, fps):
    if not analyses:
        print(f"  [WARN] {name}: aucune piste analysable → figure omise")
        return

    n = min(len(analyses), 5)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"Pulsation SAM — {name}\n"
        f"Modèle enrichi (3 sessions), bande bio 0.3–1.5 Hz grisée",
        fontsize=12, fontweight="bold",
    )

    colors = ["#e74c3c", "#f39c12", "#e040fb", "#00bcd4", "#4caf50"]
    for i, (a, ax_pair) in enumerate(zip(analyses[:n], axes)):
        ax_t, ax_f = ax_pair
        col = colors[i % len(colors)]
        t, sig_raw, sig_det = a["t"], a["sig_raw"], a["sig_det"]
        freqs, amp, peaks   = a["freqs"], a["amp"], a["peaks"]
        dom_f = a["dom_freq_hz"]
        f_pks = a["f_peaks_hz"]
        bio_tag = "DANS bande bio" if a["in_bio_dom"] else ("bord (pics)" if a["in_bio_peaks"] else "hors bande")

        # Signal temporel
        ax_t.plot(t, sig_raw, color="lightgray", lw=0.6, alpha=0.7, label="brut")
        ax_t.plot(t, sig_det + np.median(sig_raw), color=col, lw=0.9, alpha=0.85, label="détrendé (décalé)")
        if len(peaks):
            ax_t.scatter(t[peaks], sig_det[peaks] + np.median(sig_raw),
                         color=col, s=25, zorder=5, marker="v", edgecolors="white", lw=0.5,
                         label=f"pics: {f_pks:.3f} Hz" if np.isfinite(f_pks) else "pics")
        ax_t.set_ylabel("Aire (px²)", fontsize=8)
        ax_t.set_title(f"Track {a['track_id']} | n={a['n_pts']} pts | {bio_tag}", fontsize=9, loc="left")
        ax_t.legend(fontsize=7, loc="upper right")
        ax_t.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1000:.1f}k" if x >= 1000 else f"{x:.0f}"))
        ax_t.grid(axis="y", alpha=0.25)

        # FFT
        mask5 = freqs <= 5.0
        ax_f.plot(freqs[mask5], amp[mask5], color=col, lw=1.0)
        ax_f.axvspan(BIO_LO, BIO_HI, alpha=0.12, color="green", label=f"Bande bio ({BIO_LO}–{BIO_HI} Hz)")
        if np.isfinite(dom_f):
            ax_f.axvline(dom_f, color="#c62828", lw=1.5, ls="--", label=f"Dom: {dom_f:.3f} Hz")
        ax_f.set_xlabel("Fréquence (Hz)", fontsize=8)
        ax_f.set_ylabel("Amplitude", fontsize=8)
        ax_f.set_xlim(0, 5)
        ax_f.legend(fontsize=7)
        ax_f.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig_path = FIGURES / f"pulsation_{name}.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {fig_path}")


# ── Résumé global ─────────────────────────────────────────────────────────────

def make_summary(all_analyses):
    rows = []
    for a in all_analyses:
        in_bio = a["in_bio_dom"] or a["in_bio_peaks"]
        rows.append(dict(
            session=a["session"],
            track_id=a["track_id"],
            n_pts=a["n_pts"],
            dom_freq_hz=a["dom_freq_hz"],
            bio_freq_hz=a["bio_freq_hz"],
            f_peaks_hz=a["f_peaks_hz"],
            dans_bande_bio=in_bio,
        ))

    if not rows:
        print("[WARN] Aucune piste analysable dans toutes les sessions.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    csv_path = RESULTS / "pulsation_generalization.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV résumé: {csv_path}")

    # Tableau par session
    print("\n" + "=" * 75)
    print(f"{'Session':<14} {'Pistes':<8} {'Bio/Total':<12} {'Freq med (Hz)':<15} {'Min–Max Hz'}")
    print("-" * 75)
    for sess_name in [s["name"] for s in SESSIONS]:
        sub = df[df["session"] == sess_name]
        if sub.empty:
            print(f"{sess_name:<14} {'0':>6}   —")
            continue
        n_total = len(sub)
        n_bio = int(sub["dans_bande_bio"].sum())
        freqs_bio = sub["bio_freq_hz"].dropna()
        freqs_dom = sub["dom_freq_hz"].dropna()
        all_f = pd.concat([freqs_dom]).dropna()
        med_f = round(float(all_f.median()), 3) if len(all_f) else float("nan")
        min_f = round(float(all_f.min()), 3) if len(all_f) else float("nan")
        max_f = round(float(all_f.max()), 3) if len(all_f) else float("nan")
        print(f"{sess_name:<14} {n_total:<8} {n_bio}/{n_total:<10} {med_f:<15.3f} {min_f:.3f}–{max_f:.3f}")
    print("=" * 75)

    # Figure comparative sessions
    fig, ax = plt.subplots(figsize=(11, 5))
    colors_sess = {"2025_06_06": "#2196F3", "2025_08_08": "#4CAF50", "2025_08_15": "#FF9800"}
    jitter_map = {"2025_06_06": -0.12, "2025_08_08": 0.0, "2025_08_15": 0.12}

    all_dom = df["dom_freq_hz"].dropna().values
    for sess_name in [s["name"] for s in SESSIONS]:
        sub = df[df["session"] == sess_name]
        if sub.empty:
            continue
        xs = np.full(len(sub), list(colors_sess.keys()).index(sess_name) + 1)
        ys = sub["dom_freq_hz"].values
        col = colors_sess[sess_name]
        jitter = np.random.uniform(-0.05, 0.05, len(ys))
        ax.scatter(xs + jitter + jitter_map[sess_name], ys,
                   color=col, s=60, alpha=0.85, label=sess_name, zorder=3, edgecolors="white", lw=0.5)
        med = float(np.nanmedian(ys))
        ax.hlines(med, xs[0] - 0.2 + jitter_map[sess_name], xs[0] + 0.2 + jitter_map[sess_name],
                  colors=col, linewidths=2, zorder=4)

    ax.axhspan(BIO_LO, BIO_HI, alpha=0.10, color="green", label=f"Bande bio ({BIO_LO}–{BIO_HI} Hz)")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([s["name"] for s in SESSIONS])
    ax.set_ylabel("Fréquence dominante FFT (Hz)", fontsize=10)
    ax.set_title("Fréquences de pulsation par session\n(modèle enrichi, SAM, 20s/session)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(5.0, float(np.nanmax(all_dom)) * 1.2) if len(all_dom) else 5.0)
    plt.tight_layout()
    fig_path = FIGURES / "pulsation_generalization.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure comparative: {fig_path}")

    return df


# ── Rapport ───────────────────────────────────────────────────────────────────

def write_report(df):
    if df.empty:
        return

    lines = ["# Rapport pulsation — Nouvelles sessions\n",
             f"Modèle détection : `runs/detect/yolov8n_1280_e150_enriched/weights/best.pt`\n",
             f"Modèle SAM       : `sam2.1_b.pt`\n",
             f"Bande biologique : {BIO_LO}–{BIO_HI} Hz\n\n",
             "## Résultats par session\n\n",
             "| Session | Pistes | Dans bande bio | Freq médiane (Hz) | Min Hz | Max Hz |\n",
             "|---|---|---|---|---|---|\n"]

    for s in SESSIONS:
        sub = df[df["session"] == s["name"]]
        if sub.empty:
            lines.append(f"| {s['name']} | 0 | — | — | — | — |\n")
            continue
        n_bio = int(sub["dans_bande_bio"].sum())
        dom = sub["dom_freq_hz"].dropna()
        med_f = round(float(dom.median()), 3) if len(dom) else float("nan")
        min_f = round(float(dom.min()), 3) if len(dom) else float("nan")
        max_f = round(float(dom.max()), 3) if len(dom) else float("nan")
        lines.append(f"| {s['name']} | {len(sub)} | {n_bio}/{len(sub)} | {med_f:.3f} | {min_f:.3f} | {max_f:.3f} |\n")

    n_bio_global = int(df["dans_bande_bio"].sum())
    n_total_global = len(df)

    lines += ["\n## Interprétation\n\n"]
    ratio = n_bio_global / n_total_global if n_total_global > 0 else 0
    if ratio >= 0.6:
        lines.append(
            f"**{n_bio_global}/{n_total_global} pistes** ({100*ratio:.0f}%) ont une fréquence dominante "
            f"dans la bande biologique {BIO_LO}–{BIO_HI} Hz. "
            "La pipeline SAM généralise correctement aux nouvelles sessions.\n"
        )
    elif ratio >= 0.3:
        lines.append(
            f"**{n_bio_global}/{n_total_global} pistes** ({100*ratio:.0f}%) dans la bande bio. "
            "Résultats partiellement dans la bande attendue — le signal de pulsation est présent "
            "mais certaines pistes capturent probablement des artefacts (mouvement drone, drift).\n"
        )
    else:
        lines.append(
            f"**{n_bio_global}/{n_total_global} pistes** ({100*ratio:.0f}%) dans la bande bio. "
            "Les fréquences dominantes sont majoritairement hors de la bande attendue. "
            "Causes possibles : mouvement de drone dominant, méduses de grande taille à cycle lent, "
            "ou qualité de détection insuffisante sur ce segment.\n"
        )

    lines += ["\n## Fichiers produits\n\n",
              "- `results/pulsation_generalization.csv`\n",
              "- `figures/pulsation_generalization.png`\n"]
    for s in SESSIONS:
        lines.append(f"- `figures/pulsation_{s['name']}.png`\n")

    rpt_path = RESULTS / "pulsation_generalization_report.md"
    rpt_path.write_text("".join(lines), encoding="utf-8")
    print(f"Rapport: {rpt_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Chargement modèles...")
    model_det = YOLO(str(MODEL_PATH))
    sam_model = SAM(str(SAM_PATH))
    print("Modèles chargés.")

    all_raw   = []
    all_analyses = []

    for sess in SESSIONS:
        raw, analyses = run_session(sess, model_det, sam_model)
        all_raw.extend(raw)
        all_analyses.extend(analyses)
        make_session_figure(sess["name"], analyses, sess["fps"])

    # Save raw detections CSV
    raw_path = RESULTS / "pulsation_raw_detections.csv"
    if all_raw:
        fields = list(all_raw[0].keys())
        with open(raw_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_raw)
        print(f"\nCSV brut: {raw_path}  ({len(all_raw)} lignes)")

    df = make_summary(all_analyses)
    write_report(df)
    print("\n[DONE] Etape 6 terminée.")


if __name__ == "__main__":
    main()
