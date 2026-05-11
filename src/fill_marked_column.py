"""
Remplit les colonnes 'marked' et 'balisee_id' dans le CSV de trajectoires
à partir du mapping défini dans results/IDs_to_annotate_DJI0013.md.

Entrée  : results/trajectories_DJI0013_filtered.csv  (original intact)
Sortie  : results/trajectories_DJI0013_final.csv
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_FILE = ROOT / "results" / "IDs_to_annotate_DJI0013.md"
INPUT_CSV = ROOT / "results" / "trajectories_DJI0013_filtered.csv"
OUTPUT_CSV = ROOT / "results" / "trajectories_DJI0013_final.csv"


def parse_annotation(md_path: Path) -> dict[str, list[int]]:
    text = md_path.read_text(encoding="utf-8")
    section = re.search(r"## Récap final.*?(?=\n##|\Z)", text, re.DOTALL)
    if not section:
        raise ValueError("Section 'Récap final' introuvable dans le fichier MD.")
    mapping: dict[str, list[int]] = {}
    for m in re.finditer(r"Balisée #(\d+)\s*→\s*IDs\s*\[([^\]]*)\]", section.group()):
        balisee = f"Balisée #{m.group(1)}"
        ids = [int(x.strip()) for x in m.group(2).split(",") if x.strip().isdigit()]
        if ids:
            mapping[balisee] = ids
    return mapping


def main() -> None:
    mapping = parse_annotation(ANNOTATION_FILE)

    id_to_balisee: dict[int, str] = {}
    for balisee, ids in mapping.items():
        for jid in ids:
            id_to_balisee[jid] = balisee

    df = pd.read_csv(INPUT_CSV)

    df["marked"] = df["jelly_id"].apply(lambda x: 1 if x in id_to_balisee else 0)
    df["balisee_id"] = df["jelly_id"].apply(lambda x: id_to_balisee.get(x, ""))

    df.to_csv(OUTPUT_CSV, index=False)

    # ── Récap ──────────────────────────────────────────────────────────────
    total = len(df)
    n_marked = (df["marked"] == 1).sum()
    n_not_marked = total - n_marked

    print("\n=== RÉCAP FINAL ===")
    print(f"Total détections       : {total:,}")
    print(f"Détections balisées    : {n_marked:,}  ({100 * n_marked / total:.1f} %)")
    print(f"Détections non balisées: {n_not_marked:,}  ({100 * n_not_marked / total:.1f} %)")
    print()
    print("Détections par Balisée (frames) :")
    for balisee in sorted(mapping):
        sub = df[df["balisee_id"] == balisee]
        ids = mapping[balisee]
        print(f"  {balisee} : {len(sub):>5,} frames  |  IDs : {ids}")
    print()
    all_marked_ids = sorted(id_to_balisee.keys())
    print(f"IDs marqués ({len(all_marked_ids)}) : {all_marked_ids}")
    print(f"\nFichier sauvegardé : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
