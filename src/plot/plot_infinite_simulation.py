import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WANTED_COLS = [
    "web_rt","web_util","web_thr",
    "spike0_rt","spike0_util","spike0_thr",
    "mit_rt","mit_util","mit_thr",
]

def metric_label(col: str) -> str:
    if col.endswith("_rt"):
        return "Response time"
    if col.endswith("_util"):
        return "Utilization"
    if col.endswith("_thr"):
        return "Throughput"
    return "Value"

def pretty_title(col: str) -> str:
    if col.startswith("web_"):
        center = "Web"
    elif col.startswith("spike0_"):
        center = "Spike-0"
    elif col.startswith("mit_"):
        center = "Mitigation"
    else:
        center = col.split("_")[0].capitalize()
    suffix = col.split("_", 1)[1].upper()
    return f"{center} – {suffix} per batch"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to the CSV produced by export_bm_series_to_wide_csv")
    ap.add_argument("--out-dir", type=str, default="plots_infinite", help="Directory to save PNG plots")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    x = np.arange(1, len(df) + 1)

    missing = []
    generated = []

    for col in WANTED_COLS:
        if col not in df.columns:
            missing.append(col)
            continue

        y = pd.to_numeric(df[col], errors="coerce").to_numpy()
        mean_val = float(np.nanmean(y)) if y.size else float("nan")

        plt.figure()
        plt.plot(x, y)
        plt.axhline(mean_val, linestyle="--")
        plt.xlabel("Batch #")
        plt.ylabel(metric_label(col))
        plt.title(pretty_title(col))
        plt.grid(True)

        out_file = out_dir / f"{col}.png"
        plt.savefig(out_file, bbox_inches="tight", dpi=args.dpi)
        plt.close()

        generated.append(str(out_file))

    print(f"[OK] Generated {len(generated)} plots in: {out_dir}")
    if missing:
        print("[WARN] Missing columns in CSV: " + ", ".join(missing))

if __name__ == "__main__":
    main()