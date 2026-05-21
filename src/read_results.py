import pandas as pd

df = pd.read_csv(
    "C:/Users/Martinelli/Documents/GitHub/ter_meduses/runs/detect/yolov8n_1280_e150_enriched/results.csv"
)
df.columns = df.columns.str.strip()

col_map50   = [c for c in df.columns if "mAP50" in c and "95" not in c][0]
col_map5095 = [c for c in df.columns if "mAP50-95" in c][0]
col_prec    = [c for c in df.columns if "precision" in c.lower()][0]
col_rec     = [c for c in df.columns if "recall" in c.lower()][0]
col_epoch   = "epoch"

best_idx = df[col_map50].idxmax()
best = df.iloc[best_idx]

print(f"Epochs completes : {len(df)}")
print(f"Meilleure epoch  : {int(best[col_epoch])}")
print(f"mAP50 (best)     : {best[col_map50]:.4f}")
print(f"mAP50-95 (best)  : {best[col_map5095]:.4f}")
print(f"Precision (best) : {best[col_prec]:.4f}")
print(f"Recall    (best) : {best[col_rec]:.4f}")
print()
print(f"-- Derniere epoch ({int(df.iloc[-1][col_epoch])}) --")
print(f"mAP50            : {df.iloc[-1][col_map50]:.4f}")
print(f"Precision        : {df.iloc[-1][col_prec]:.4f}")
print(f"Recall           : {df.iloc[-1][col_rec]:.4f}")
