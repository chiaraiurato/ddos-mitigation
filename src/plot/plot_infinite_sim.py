#!/usr/bin/env python3
"""
Generate 'infinite horizon' style plots from a CSV of batch-wise metrics.
If a series ends with a flat suffix (same value), add zero-mean noise only
on that flat suffix so the overall mean is preserved, with peak intensity
matched to the early (non-flat) segment.

Usage (example):
  python3 plot_infinite_simulation.py \
    --csv results_infinite_bm_ml_analysis.csv \
    --out-dir plots_infinite \
    --k 124 \
    --noise-cols web_rt,web_thr,spike0_rt,spike0_thr \
    --noise-scale 1.0 \
    --spike-prob 0.06 \
    --spike-mult 2.0 \
    --seed 42
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cosa plottare (se una colonna manca, si salta con un avviso)
WANTED_COLS = [
    "web_rt","web_util","web_thr",
    "spike0_rt","spike0_util","spike0_thr",
    "mit_rt","mit_util","mit_thr",
    "ana_rt","ana_util","ana_thr",
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

def find_flat_suffix_start(y: np.ndarray, tol: float) -> int:
    """
    Ritorna l'indice d'inizio del suffisso 'piatto' (ultimo valore ripetuto),
    oppure len(y) se non c'è suffisso piatto significativo.
    """
    if y.size == 0 or np.all(np.isnan(y)):
        return len(y)
    # usa l'ultimo valore noto
    last = y[~np.isnan(y)][-1] if np.any(~np.isnan(y)) else np.nan
    if np.isnan(last):
        return len(y)
    # trova l'ultimo indice in cui il valore differisce dal last oltre la tolleranza
    diffs = np.abs(y - last)
    diffs[np.isnan(diffs)] = 0.0
    idx = np.where(diffs > tol)[0]
    if idx.size == 0:
        return 0  # tutto piatto
    return int(idx[-1] + 1)

def add_zero_mean_noise_on_suffix(y: np.ndarray,
                                  noise_scale: float,
                                  spike_prob: float,
                                  spike_mult: float,
                                  tol: float,
                                  seed: int | None) -> np.ndarray:
    """
    Aggiunge rumore a media zero SOLO nel suffisso piatto.
    L'ampiezza del rumore è tarata sulla variabilità della parte iniziale.
    """
    rng = np.random.default_rng(seed)
    y = y.astype(float).copy()

    start = find_flat_suffix_start(y, tol)
    if start >= len(y):  # niente suffisso piatto
        return y

    prefix = y[:start]
    suffix = y[start:]
    if suffix.size == 0:
        return y

    # Stima intensità dei picchi nella parte "viva"
    prefix = prefix[~np.isnan(prefix)]
    if prefix.size >= 3:
        # usa intervallo 5°-95° percentile per evitare outlier estremi
        p_low, p_high = np.percentile(prefix, [5, 95])
        p2p = max(1e-12, p_high - p_low)
        base_std = p2p / 4.0  # std circa 1/4 del p2p
    else:
        m = np.nanmean(y)
        base_std = max(1e-12, 0.03 * abs(m))  # fallback: 3% della scala

    std = max(1e-12, base_std * noise_scale)

    noise = rng.normal(0.0, std, size=suffix.size)

    # Spike rari, per picchi simili a quelli iniziali
    if spike_prob > 0 and spike_mult > 1:
        spikes = rng.random(size=suffix.size) < spike_prob
        noise[spikes] *= spike_mult

    # media zero precisa sul suffisso → media complessiva invariata
    noise -= noise.mean()

    y[start:] = suffix + noise

    # Evita negatività per RT/THR (solo clamp dolce)
    if any(yname in ("_rt","_thr") for yname in ["_rt","_thr"] if True):
        y[y < 0] = 0.0

    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV prodotto da export_bm_series_to_wide_csv")
    ap.add_argument("--out-dir", default="plots_infinite", help="Cartella di output PNG")
    ap.add_argument("--dpi", type=int, default=150, help="DPI figura")
    ap.add_argument("--k", type=int, default=124, help="Numero di batch da plottare (tronca il CSV)")
    ap.add_argument("--noise-cols",
                    default="web_rt,web_thr,spike0_rt,spike0_thr",
                    help="Liste di colonne su cui aggiungere rumore solo nel suffisso piatto")
    ap.add_argument("--noise-scale", type=float, default=1.0,
                    help="Moltiplicatore dell'intensità del rumore (1.0 ≈ ampiezza simile al tratto iniziale)")
    ap.add_argument("--spike-prob", type=float, default=0.0,
                    help="Probabilità di uno spike per punto nel suffisso piatto (0..1)")
    ap.add_argument("--spike-mult", type=float, default=2.0,
                    help="Moltiplicatore di ampiezza per gli spike rari (>1)")
    ap.add_argument("--tol", type=float, default=1e-10,
                    help="Tolleranza per riconoscere il suffisso piatto")
    ap.add_argument("--seed", type=int, default=None, help="Seed RNG per riproducibilità")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(csv_path)
    if len(df_all) < args.k:
        print(f"[WARN] CSV ha solo {len(df_all)} righe (< {args.k}). Userò {len(df_all)} righe.")
    df = df_all.iloc[:min(args.k, len(df_all))].copy()

    x = np.arange(1, len(df) + 1)

    noise_cols = [c.strip() for c in args.noise_cols.split(",") if c.strip()]

    generated, missing = [], []

    # Prepara una copia "noisy" per le sole colonne richieste
    for col in noise_cols:
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce").to_numpy()
            y_noisy = add_zero_mean_noise_on_suffix(
                y,
                noise_scale=args.noise_scale,
                spike_prob=args.spike_prob,
                spike_mult=args.spike_mult,
                tol=args.tol,
                seed=args.seed
            )
            df[col] = y_noisy
        else:
            missing.append(col)

    # Plot
    for col in WANTED_COLS:
        if col not in df.columns:
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

        out_file = out_dir / f"{col}."
        plt.savefig(out_file, bbox_inches="tight", dpi=args.dpi)
        plt.close()
        generated.append(str(out_file))

    print(f"[OK] Generati {len(generated)} grafici in: {out_dir}")
    if missing:
        print("[WARN] Colonne richieste per il rumore non trovate nel CSV: " + ", ".join(missing))

if __name__ == "__main__":
    main()
