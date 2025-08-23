"""
Generate 'infinite horizon' style plots from a CSV of batch-wise metrics.
- Uses exactly the first K batches (default K=128).
- Requires at least K rows in the CSV.
- For each metric, plots only if it has at least K non-NaN values in the first K rows.
- Overlays the mean across the first K batches as a dashed horizontal line.

Usage:
  python3 plot_infinite_simulation.py \
    --csv results_infinite_bm.csv \
    --out-dir infinite_horizon \
    --dpi 150 \
    --k 128
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Default set; 'ana_*' verranno inclusi automaticamente se presenti nel CSV
DEFAULT_WANTED_COLS = [
    "web_rt","web_util","web_thr",
    "spike0_rt","spike0_util","spike0_thr",
    "mit_rt","mit_util","mit_thr",
    # "ana_rt","ana_util","ana_thr"  # aggiunte dinamicamente se presenti
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
    elif col.startswith("ana_"):
        center = "Analysis"
    else:
        center = col.split("_")[0].capitalize()
    suffix = col.split("_", 1)[1].upper()
    return f"{center} – {suffix} per batch"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True,
                    help="Path to the CSV produced by export_bm_series_to_wide_csv")
    ap.add_argument("--out-dir", type=str, default="plots_infinite",
                    help="Directory to save PNG plots")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    ap.add_argument("--k", type=int, default=128,
                    help="Number of batches to plot (requires at least this many rows)")
    ap.add_argument("--cols", type=str, default="",
                    help="Comma-separated column list to plot; if empty, use defaults + ana_* if present")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Enforce at least K rows in the CSV
    n_rows = len(df)
    if n_rows < args.k:
        raise SystemExit(
            f"ERROR: CSV has only {n_rows} rows; need at least {args.k} batches.\n"
            f"→ Rerun the infinite-horizon simulation to generate ≥ {args.k} batch means."
        )

    # Slice exactly the first K batches
    df_k = df.iloc[:args.k].copy()
    x = np.arange(1, args.k + 1)

    # Decide columns to plot
    if args.cols.strip():
        wanted_cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    else:
        wanted_cols = list(DEFAULT_WANTED_COLS)
        # Add ANA columns if present in CSV
        for c in ("ana_rt", "ana_util", "ana_thr"):
            if c in df.columns:
                wanted_cols.append(c)

    missing_cols = []
    skipped_insufficient = []
    generated = []

    for col in wanted_cols:
        if col not in df_k.columns:
            missing_cols.append(col)
            continue

        y = pd.to_numeric(df_k[col], errors="coerce").to_numpy()
        valid_mask = np.isfinite(y)
        valid_count = int(valid_mask.sum())

        # Require K valid points (no NaN) in the first K batches for this column
        if valid_count < args.k:
            skipped_insufficient.append(f"{col} (valid {valid_count}/{args.k})")
            continue

        mean_val = float(np.nanmean(y))

        plt.figure()
        plt.plot(x, y)  # one chart per figure, do not set colors
        plt.axhline(mean_val, linestyle="--")
        plt.xlabel("Batch #")
        plt.ylabel(metric_label(col))
        plt.title(pretty_title(col))
        plt.grid(True)

        out_file = out_dir / f"{col}.png"
        plt.savefig(out_file, bbox_inches="tight", dpi=args.dpi)
        plt.close()
        generated.append(str(out_file))

    # Summary
    print(f"[OK] Generated {len(generated)} plots in: {out_dir}")
    if missing_cols:
        print("[WARN] Missing columns in CSV: " + ", ".join(missing_cols))
    if skipped_insufficient:
        print("[WARN] Skipped (not enough valid points in first K rows): "
              + ", ".join(skipped_insufficient))

# python3 plot_infinite_horizon_simulation.py --csv results_infinite_bm_ml_analysis.csv --out-dir plots_infinite_improved --dpi 150 --k 128
if __name__ == "__main__":
    main()
