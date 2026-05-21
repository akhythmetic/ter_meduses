"""
Application d'annotation manuelle des pulsations de meduses (Step 3).

Interface Tkinter :
  - Lecteur video (chaque clip 10s, 800x800, 29.97 fps)
  - Compteur de pulsations (clic bouton ou touche ESPACE)
  - Jugement binaire : Pulse / Ne pulse pas / Incertain
  - Notes libres par fenetre
  - Sauvegarde automatique apres chaque fenetre
  - Navigation : Suivant / Precedent
  - Resume possible si annotations_observerN.csv existe deja

Usage:
    python src/annotation_app.py --observer 1
    python src/annotation_app.py --observer 2 --clips results/clips
"""

import argparse
import csv
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

REPO      = Path(__file__).resolve().parent.parent
CLIPS_DIR = REPO / "results" / "clips"
CSV_DIR   = REPO / "results"

FPS         = 29.97
DISPLAY_W   = 800
DISPLAY_H   = 800
PLAY_DELAY  = int(1000 / FPS)  # ms entre frames


class AnnotationApp:
    def __init__(self, root: tk.Tk, clips: list[Path], out_csv: Path) -> None:
        self.root     = root
        self.clips    = clips
        self.out_csv  = out_csv
        self.n        = len(clips)

        # Charge annotations existantes pour reprise
        self.annotations: dict[str, dict] = {}
        if out_csv.exists():
            with open(out_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.annotations[row["window_id"]] = row

        self.idx          = 0
        self.pulse_count  = 0
        self.playing      = False
        self.cap: cv2.VideoCapture | None = None
        self._after_id: str | None = None

        self._build_ui()
        self._jump_to_first_unannotated()
        self._load_clip(self.idx)

    # ── UI ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.title("Annotation pulsations meduses")
        self.root.resizable(False, False)

        # Header info
        top = tk.Frame(self.root, pady=4)
        top.pack(fill="x")
        self.lbl_progress = tk.Label(top, text="", font=("Arial", 11, "bold"))
        self.lbl_progress.pack(side="left", padx=10)
        self.lbl_window = tk.Label(top, text="", font=("Arial", 11))
        self.lbl_window.pack(side="left", padx=10)

        # Video canvas
        self.canvas = tk.Canvas(self.root, width=DISPLAY_W, height=DISPLAY_H, bg="black")
        self.canvas.pack()

        # Lecteur controls
        ctrl = tk.Frame(self.root, pady=4)
        ctrl.pack(fill="x")
        self.btn_play = tk.Button(ctrl, text="Play", width=8, command=self._toggle_play)
        self.btn_play.pack(side="left", padx=6)
        tk.Button(ctrl, text="Rewind", width=8, command=self._rewind).pack(side="left", padx=4)
        self.lbl_time = tk.Label(ctrl, text="0.0s / 10.0s", width=14)
        self.lbl_time.pack(side="left", padx=6)

        # Compteur pulsations
        pf = tk.LabelFrame(self.root, text="Pulsations observees", pady=6, padx=10)
        pf.pack(fill="x", padx=10, pady=4)
        self.lbl_count = tk.Label(pf, text="0", font=("Arial", 36, "bold"), fg="#1a6e9e")
        self.lbl_count.pack(side="left", padx=10)
        tk.Button(pf, text="+ Pulsation  [ESPACE]", font=("Arial", 13),
                  command=self._add_pulse, height=2, width=22).pack(side="left", padx=10)
        tk.Button(pf, text="- Annuler", font=("Arial", 12),
                  command=self._remove_pulse, height=2, width=10).pack(side="left", padx=4)

        # Jugement
        jf = tk.LabelFrame(self.root, text="Jugement global", pady=6, padx=10)
        jf.pack(fill="x", padx=10, pady=4)
        self.judgment_var = tk.StringVar(value="")
        for val, lbl, color in [
            ("pulse",         "Pulse",        "#2d8a2d"),
            ("no_pulse",      "Ne pulse pas", "#b22222"),
            ("uncertain",     "Incertain",    "#b8860b"),
        ]:
            tk.Radiobutton(
                jf, text=lbl, variable=self.judgment_var, value=val,
                font=("Arial", 12), fg=color, selectcolor="#f0f0f0"
            ).pack(side="left", padx=14)

        # Notes
        nf = tk.LabelFrame(self.root, text="Notes (optionnel)", pady=4, padx=10)
        nf.pack(fill="x", padx=10, pady=4)
        self.notes_entry = tk.Entry(nf, font=("Arial", 11), width=60)
        self.notes_entry.pack(fill="x", padx=4)

        # Navigation
        nav = tk.Frame(self.root, pady=6)
        nav.pack(fill="x")
        tk.Button(nav, text="< Precedent", width=12, command=self._prev).pack(side="left", padx=10)
        tk.Button(nav, text="Suivant >", width=12, command=self._next,
                  bg="#4a90d9", fg="white", font=("Arial", 11, "bold")).pack(side="right", padx=10)

        # Raccourcis clavier
        self.root.bind("<space>",  lambda e: self._add_pulse())
        self.root.bind("<Return>", lambda e: self._next())
        self.root.bind("<Left>",   lambda e: self._prev())
        self.root.bind("<Right>",  lambda e: self._next())

    # ── Navigation ──────────────────────────────────────────────────────

    def _jump_to_first_unannotated(self) -> None:
        for i, clip in enumerate(self.clips):
            wid = clip.stem
            if wid not in self.annotations:
                self.idx = i
                return
        self.idx = self.n - 1  # toutes annotees : reste sur la derniere

    def _load_clip(self, idx: int) -> None:
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.playing = False
        self.btn_play.config(text="Play")

        if self.cap:
            self.cap.release()

        path = self.clips[idx]
        self.cap = cv2.VideoCapture(str(path))

        # Restore annotation si existe
        wid = path.stem
        ann = self.annotations.get(wid, {})
        self.pulse_count = int(ann.get("n_pulsations_observees", 0))
        self.judgment_var.set(ann.get("jugement", ""))
        self.notes_entry.delete(0, tk.END)
        self.notes_entry.insert(0, ann.get("notes", ""))

        self.lbl_count.config(text=str(self.pulse_count))
        self.lbl_progress.config(text=f"[{idx+1}/{self.n}]")
        self.lbl_window.config(text=wid)

        self._show_frame()

    def _prev(self) -> None:
        self._save_current()
        if self.idx > 0:
            self.idx -= 1
            self._load_clip(self.idx)

    def _next(self) -> None:
        if not self._validate():
            return
        self._save_current()
        if self.idx < self.n - 1:
            self.idx += 1
            self._load_clip(self.idx)
        else:
            messagebox.showinfo("Termine", f"Toutes les fenetres annotees.\nFichier : {self.out_csv.name}")

    # ── Lecture video ────────────────────────────────────────────────────

    def _toggle_play(self) -> None:
        if self.playing:
            self.playing = False
            self.btn_play.config(text="Play")
            if self._after_id:
                self.root.after_cancel(self._after_id)
                self._after_id = None
        else:
            self.playing = True
            self.btn_play.config(text="Pause")
            self._play_loop()

    def _play_loop(self) -> None:
        if not self.playing or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.playing = False
            self.btn_play.config(text="Play")
            return
        self._render_frame(frame)
        pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.lbl_time.config(text=f"{pos_ms/1000:.1f}s / 10.0s")
        self._after_id = self.root.after(PLAY_DELAY, self._play_loop)

    def _rewind(self) -> None:
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.playing = False
        self.btn_play.config(text="Play")
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.lbl_time.config(text="0.0s / 10.0s")
        self._show_frame()

    def _show_frame(self) -> None:
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            self._render_frame(frame)

    def _render_frame(self, frame: "cv2.Mat") -> None:
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=photo)
        self.canvas.image = photo  # keep reference

    # ── Compteur ────────────────────────────────────────────────────────

    def _add_pulse(self) -> None:
        self.pulse_count += 1
        self.lbl_count.config(text=str(self.pulse_count))

    def _remove_pulse(self) -> None:
        if self.pulse_count > 0:
            self.pulse_count -= 1
            self.lbl_count.config(text=str(self.pulse_count))

    # ── Sauvegarde ──────────────────────────────────────────────────────

    def _validate(self) -> bool:
        if not self.judgment_var.get():
            messagebox.showwarning("Jugement manquant",
                                   "Selectionnez un jugement (Pulse / Ne pulse pas / Incertain).")
            return False
        return True

    def _save_current(self) -> None:
        wid = self.clips[self.idx].stem
        self.annotations[wid] = {
            "window_id":              wid,
            "n_pulsations_observees": self.pulse_count,
            "jugement":               self.judgment_var.get(),
            "notes":                  self.notes_entry.get(),
            "timestamp":              time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._flush_csv()

    def _flush_csv(self) -> None:
        fieldnames = ["window_id", "n_pulsations_observees", "jugement", "notes", "timestamp"]
        with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for wid in [c.stem for c in self.clips]:
                if wid in self.annotations:
                    writer.writerow(self.annotations[wid])

    def on_close(self) -> None:
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ── Main ──────────────────────────────────────────────────────────────

def main(observer: int, clips_dir: Path) -> None:
    clips = sorted(clips_dir.glob("W*.mp4"))
    if not clips:
        print(f"Aucun clip trouve dans {clips_dir}")
        return

    out_csv = CSV_DIR / f"annotations_observer{observer}.csv"
    print(f"Observer {observer} | {len(clips)} clips | sortie : {out_csv.name}")

    root = tk.Tk()
    app  = AnnotationApp(root, clips, out_csv)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observer", type=int, default=1,
                        help="Numero de l'observateur (1, 2, ...)")
    parser.add_argument("--clips",    type=Path, default=CLIPS_DIR,
                        help="Dossier contenant les clips .mp4")
    args = parser.parse_args()
    main(args.observer, args.clips)
