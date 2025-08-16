"""
Spaghetti plot per analisi a ORIZZONTE FINITO (senza legenda) + tabella QoS.
- Legge il CSV dei checkpoint (es. results_finite_simulation.csv)
- Salva grafici in ./finite_simulation/
- Genera anche ./finite_simulation/qos_summary.csv e .md

Esempio:
python3 plot_finite_simulation.py --csv results_finite_simulation.csv --time-unit hour --vline 10800 \
  --qos-min-legal-completed 0.70 --qos-max-illegal-completed 0.15
"""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Colonne
TIME_COL = "time"
REP_COL  = "replica"
FINAL_COL= "is_final"

# Metriche (plottiamo solo quelle effettivamente presenti nel CSV)
METRICS = [
    "web_rt_mean", "mit_rt_mean", "system_rt_mean",   # << aggiunta qui
    "web_util", "web_throughput",
    "mit_util", "mit_throughput",
    "spikes_count",
    "illegal_share",
    "processed_legal_share", "processed_illegal_share",
    "completed_legal_share", "completed_illegal_share",
    "completed_legal_of_completed_share", "completed_illegal_of_completed_share",
]

def metric_to_label(metric: str) -> str:
    if metric == "system_rt_mean":     return "Tempo di risposta medio globale (s)"  # << nuovo
    if metric.endswith("rt_mean"):     return "Tempo di risposta medio (s)"
    if metric.endswith("throughput"):  return "Throughput (jobs/s)"
    if metric.endswith("util"):        return "Utilizzazione (%)"
    if metric == "spikes_count":       return "Spike allocati"
    if "share" in metric:              return "Percentuale (%)"
    return metric


# quali metriche sono percentuali (0..1) e vanno mostrate in %
PERCENT_METRICS = {
    "web_util", "mit_util",
    "illegal_share",
    "processed_legal_share", "processed_illegal_share",
    "completed_legal_share", "completed_illegal_share",
    "completed_legal_of_completed_share", "completed_illegal_of_completed_share",
}

# ---------- helpers ----------
def guess_available_metrics(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [m for m in candidates if m in df.columns]

def time_scale_and_label(unit: str):
    unit = unit.lower()
    if unit in ("s","sec","second","seconds"): return 1.0, "Tempo (s)"
    if unit in ("h","hour","hours"):           return 1/3600.0, "Tempo (h)"
    if unit in ("d","day","days"):             return 1/86400.0, "Tempo (giorni)"
    raise ValueError("time-unit non riconosciuta: usa s|h|hour|day")

# ---------- plotting ----------
def spaghetti_no_legend(df: pd.DataFrame,
                        metric: str,
                        scale: float,
                        xlabel: str,
                        outdir: Path,
                        round_time_decimals: Optional[int] = 0,
                        vline_sec: Optional[float] = None):
    d = df[[TIME_COL, REP_COL, metric]].dropna().copy()
    if d.empty:
        return None

    if round_time_decimals is not None:
        d[TIME_COL] = d[TIME_COL].round(round_time_decimals)
    d = d.sort_values([REP_COL, TIME_COL])

    plt.figure()
    ax = plt.gca()

    for _, g in d.groupby(REP_COL):
        gx = g[TIME_COL].to_numpy() * scale
        gy = g[metric].to_numpy()

        # portiamo in percentuale le metriche percentuali
        if metric in PERCENT_METRICS:
            gy = gy * 100.0

        plt.plot(gx, gy, linewidth=1.2)

    if vline_sec is not None:
        plt.axvline(x=vline_sec * scale, linestyle="--")

    plt.xlabel(xlabel)
    plt.ylabel(metric_to_label(metric))
    plt.title(f"{metric} — orizzonte finito")
    plt.grid(True)

    # formattazione asse Y per percentuali
    if metric in PERCENT_METRICS:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        # limiti "comodi" 0..100 se util/share
        ymin, ymax = ax.get_ylim()
        lo = 0
        hi = min(max(100, ymax), 100) if "util" in metric or "share" in metric else ymax
        ax.set_ylim(lo, hi)

    plt.tight_layout()
    outpath = outdir / f"{metric}_spaghetti.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    return outpath

def qos_tables(df: pd.DataFrame,
               outdir: Path,
               min_legal_completed: Optional[float],
               max_illegal_completed: Optional[float]):

    # prendiamo solo i checkpoint finali per replica
    if FINAL_COL in df.columns:
        dfin = df[df[FINAL_COL] == True].copy()
        if dfin.empty:
            # fallback: ultimo tempo per replica
            dfin = df.sort_values(TIME_COL).groupby(REP_COL, as_index=False).tail(1)
    else:
        dfin = df.sort_values(TIME_COL).groupby(REP_COL, as_index=False).tail(1)

    keep_cols = [
        REP_COL, TIME_COL,
        "completed_legal_share", "completed_illegal_share",
        "processed_legal_share", "processed_illegal_share",
        "illegal_share"
    ]
    keep_cols = [c for c in keep_cols if c in dfin.columns]
    d = dfin[keep_cols].copy()

    # convertiamo in %
    for c in d.columns:
        if c.endswith("_share"):
            d[c] = d[c] * 100.0

    # QoS check (opzionali): PASS/FAIL
    if ("completed_legal_share" in d.columns) and (min_legal_completed is not None):
        d["qos_min_legal_completed_pass"] = d["completed_legal_share"] >= (min_legal_completed * 100.0)
    if ("completed_illegal_share" in d.columns) and (max_illegal_completed is not None):
        d["qos_max_illegal_completed_pass"] = d["completed_illegal_share"] <= (max_illegal_completed * 100.0)

    # salva CSV
    csv_out = outdir / "qos_summary.csv"
    d.to_csv(csv_out, index=False)

    # salva markdown breve
    md_out = outdir / "qos_summary.md"
    with open(md_out, "w") as f:
        f.write("| replica | time | compl_legal% | compl_illegal% | proc_legal% | proc_illegal% | illegal_arrivals% | QoS |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|:--:|\n")
        for _, r in d.iterrows():
            qos_flag = "—"
            flags = []
            if "qos_min_legal_completed_pass" in r:
                flags.append("L OK" if r["qos_min_legal_completed_pass"] else "L FAIL")
            if "qos_max_illegal_completed_pass" in r:
                flags.append("I OK" if r["qos_max_illegal_completed_pass"] else "I FAIL")
            if flags: qos_flag = ", ".join(flags)

            def fmt(c): 
                return f"{r[c]:.2f}" if c in d.columns else ""
            f.write(f"| {int(r[REP_COL])} | {r[TIME_COL]:.0f} | {fmt('completed_legal_share')} | {fmt('completed_illegal_share')} | {fmt('processed_legal_share')} | {fmt('processed_illegal_share')} | {fmt('illegal_share')} | {qos_flag} |\n")

    return csv_out, md_out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Grafici orizzonte finito (spaghetti, senza legenda) + QoS")
    ap.add_argument("--csv", type=str, default="results_finite_simulation.csv",
                    help="Percorso CSV (checkpoint orizzonte finito)")
    ap.add_argument("--time-unit", type=str, default="h",
                    choices=["s","h","hour","day"],
                    help="Unità asse X: s|h|hour|day")
    ap.add_argument("--round-time", type=int, default=0,
                    help="Arrotondamento tempo (decimali). -1 per disattivare")
    ap.add_argument("--vline", type=float, default=None,
                    help="Linea verticale a t (in secondi), es. 10800 per 3 ore")
    ap.add_argument("--qos-min-legal-completed", type=float, default=None,
                    help="Soglia minima per completed_legal_share (0..1) per PASS")
    ap.add_argument("--qos-max-illegal-completed", type=float, default=None,
                    help="Soglia massima per completed_illegal_share (0..1) per PASS")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir   = Path("finite_simulation")
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # metriche disponibili
    metrics = guess_available_metrics(df, METRICS)
    if not metrics:
        raise ValueError(f"Nessuna metrica trovata tra {METRICS}. Colonne presenti: {list(df.columns)}")

    # scala temporale + arrotondamento
    scale, xlabel = time_scale_and_label(args.time_unit)
    round_time_decimals = None if args.round_time is not None and args.round_time < 0 else args.round_time

    # genera spaghetti plot SENZA legenda
    for metric in metrics:
        spaghetti_no_legend(df, metric, scale, xlabel, outdir,
                            round_time_decimals=round_time_decimals, vline_sec=args.vline)

    # tabella QoS finale
    qos_tables(
        df,
        outdir,
        min_legal_completed=args.qos_min_legal_completed,
        max_illegal_completed=args.qos_max_illegal_completed
    )

    print(f"[OK] Grafici e tabelle salvati in: {outdir.resolve()}")

if __name__ == "__main__":
    main()
