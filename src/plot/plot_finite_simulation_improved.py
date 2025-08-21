"""
Spaghetti plot per analisi a ORIZZONTE FINITO (senza legenda) + tabella QoS.
- Legge il CSV dei checkpoint (es. results_finite_simulation.csv)
- Salva grafici in ./finite_simulation/
- Genera anche ./finite_simulation/qos_summary.csv e .md
"""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Colonne
TIME_COL = "time"
REP_COL  = "replica"
FINAL_COL= "is_final"

# Metriche base (le per-spike verranno aggiunte dinamicamente dal CSV)
# + aggiunte le metriche del centro Analysis ML
BASE_METRICS = [
    "web_rt_mean", "mit_rt_mean", "system_rt_mean",
    "web_util", "web_throughput",
    "mit_util", "mit_throughput",
    # --- Analysis ML (nuove) ---
    "analysis_rt_mean", "analysis_util", "analysis_throughput",
    # ---
    "spikes_count",
    "illegal_share",
    "processed_legal_share", "processed_illegal_share",
    "completed_legal_share", "completed_illegal_share",
    "completed_legal_of_completed_share", "completed_illegal_of_completed_share",
]

def metric_to_label(metric: str) -> str:
    # etichette asse Y
    if metric == "system_rt_mean":     return "Tempo di risposta medio globale (s)"
    if metric.endswith("rt_mean"):     return "Tempo di risposta medio (s)"
    if metric.endswith("throughput"):  return "Throughput (jobs/s)"
    if metric.endswith("util"):        return "Utilizzazione"
    if metric == "spikes_count":       return "Spike allocati"
    if "share" in metric:              return "Percentuale (%)"
    return metric

# metriche “share” (0..1) da mostrare come percentuali 0..100
PERCENT_METRICS = {
    "illegal_share",
    "processed_legal_share", "processed_illegal_share",
    "completed_legal_share", "completed_illegal_share",
    "completed_legal_of_completed_share", "completed_illegal_of_completed_share",
}

# ---------- helpers ----------
def guess_available_metrics(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [m for m in candidates if m in df.columns]

def find_spike_metrics(df: pd.DataFrame) -> List[str]:
    """
    Cerca automaticamente colonne per-spike del tipo:
      spike{i}_rt_mean, spike{i}_util, spike{i}_throughput
    """
    cols = []
    for c in df.columns:
        if not c.startswith("spike"):
            continue
        if c.endswith("_rt_mean") or c.endswith("_util") or c.endswith("_throughput"):
            cols.append(c)
    # ordinamento “umano”: spike0_..., spike1_..., ecc.
    def key(c):
        # estrae l’indice numerico dopo 'spike'
        try:
            left = c.split("_", 1)[0]  # spike0
            i = int(left.replace("spike", ""))
        except Exception:
            i = 10**9
        # ordina per (indice, tipo metrica)
        if c.endswith("_rt_mean"): suf = 0
        elif c.endswith("_util"): suf = 1
        else: suf = 2  # throughput
        return (i, suf, c)
    return sorted(cols, key=key)

def time_scale_and_label(unit: str):
    unit = unit.lower()
    if unit in ("s","sec","second","seconds"): return 1.0, "Tempo (s)"
    if unit in ("h","hour","hours"):           return 1/3600.0, "Tempo (h)"
    if unit in ("d","day","days"):             return 1/86400.0, "Tempo (giorni)"
    raise ValueError("time-unit non riconosciuta: usa s|h|hour|day")

# ---------- plotting generico (no util) ----------
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
        if metric in PERCENT_METRICS:
            gy = gy * 100.0  # mostriamo in percentuale solo le “share”
        plt.plot(gx, gy, linewidth=1.2)

    if vline_sec is not None:
        plt.axvline(x=vline_sec * scale, linestyle="--")

    plt.xlabel(xlabel)
    plt.ylabel(metric_to_label(metric))
    plt.title(f"{metric} — orizzonte finito")
    plt.grid(True)
    plt.tight_layout()

    outpath = outdir / f"{metric}_spaghetti.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    return outpath

# ---------- plotting dedicato alle UTIL ----------
def spaghetti_util(df: pd.DataFrame,
                   metric: str,
                   scale: float,
                   xlabel: str,
                   outdir: Path,
                   round_time_decimals: Optional[int] = 0,
                   vline_sec: Optional[float] = None):
    """Plot 0..1 con headroom sopra 1.0 e tick a 3 decimali (web/mit/spike*)."""
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
        # niente conversione in percentuale: util resta [0..1] float
        plt.plot(gx, gy, linewidth=1.2)

    if vline_sec is not None:
        plt.axvline(x=vline_sec * scale, linestyle="--")

    headroom = 0.001  # ~0.1% sopra 1.0
    ax.set_ylim(0.0, 1.0 + headroom)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.axhline(1.0, color='gray', linewidth=0.8, alpha=0.6)

    plt.xlabel(xlabel)
    plt.ylabel(metric_to_label(metric))
    plt.title(f"{metric} — orizzonte finito")
    plt.grid(True)
    plt.tight_layout()

    outpath = outdir / f"{metric}_spaghetti.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    return outpath

def spaghetti_util_zoom(df: pd.DataFrame,
                        metric: str,
                        scale: float,
                        xlabel: str,
                        outdir: Path,
                        round_time_decimals: Optional[int] = 0,
                        vline_sec: Optional[float] = None):
    """Zoom automatico intorno ai valori osservati, con headroom e 3 decimali."""
    d = df[[TIME_COL, REP_COL, metric]].dropna().copy()
    if d.empty:
        return None

    if round_time_decimals is not None:
        d[TIME_COL] = d[TIME_COL].round(round_time_decimals)
    d = d.sort_values([REP_COL, TIME_COL])

    ymin_obs = d[metric].min()
    ymax_obs = d[metric].max()
    spread = max(ymax_obs - ymin_obs, 1e-6)
    delta  = max(0.0003, spread * 0.30)  # margine dinamico
    low  = max(0.0, ymin_obs - delta)
    high = min(1.0 + 0.001, ymax_obs + delta)  # un filo sopra 1.0

    plt.figure()
    ax = plt.gca()
    for _, g in d.groupby(REP_COL):
        gx = g[TIME_COL].to_numpy() * scale
        gy = g[metric].to_numpy()
        plt.plot(gx, gy, linewidth=1.2)

    if vline_sec is not None:
        plt.axvline(x=vline_sec * scale, linestyle="--")

    ax.set_ylim(low, high)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.axhline(1.0, color='gray', linewidth=0.8, alpha=0.6)

    plt.xlabel(xlabel)
    plt.ylabel(metric_to_label(metric))
    plt.title(f"{metric} — orizzonte finito (zoom)")
    plt.grid(True)
    plt.tight_layout()

    outpath = outdir / f"{metric}_spaghetti_zoom.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    return outpath

# ---------- QoS ----------
def qos_tables(df: pd.DataFrame,
               outdir: Path,
               min_legal_completed: Optional[float],
               max_illegal_completed: Optional[float]):

    if FINAL_COL in df.columns:
        dfin = df[df[FINAL_COL] == True].copy()
        if dfin.empty:
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

    # quote in percentuale
    for c in d.columns:
        if c.endswith("_share"):
            d[c] = d[c] * 100.0

    if ("completed_legal_share" in d.columns) and (min_legal_completed is not None):
        d["qos_min_legal_completed_pass"] = d["completed_legal_share"] >= (min_legal_completed * 100.0)
    if ("completed_illegal_share" in d.columns) and (max_illegal_completed is not None):
        d["qos_max_illegal_completed_pass"] = d["completed_illegal_share"] <= (max_illegal_completed * 100.0)

    csv_out = outdir / "qos_summary.csv"
    d.to_csv(csv_out, index=False)

    md_out = outdir / "qos_summary.md"
    with open(md_out, "w") as f:
        f.write("| replica | time | compl_legal% | compl_illegal% | proc_legal% | proc_illegal% | illegal_arrivals% | QoS |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|:--:|\n")
        for _, r in d.iterrows():
            flags = []
            if "qos_min_legal_completed_pass" in r:
                flags.append("L OK" if r["qos_min_legal_completed_pass"] else "L FAIL")
            if "qos_max_illegal_completed_pass" in r:
                flags.append("I OK" if r["qos_max_illegal_completed_pass"] else "I FAIL")
            qos_flag = "—" if not flags else ", ".join(flags)

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
    outdir   = Path("finite_simulation_improved")
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # metriche di base + metriche per-spike dal CSV
    metrics_base  = guess_available_metrics(df, BASE_METRICS)
    metrics_spike = find_spike_metrics(df)
    metrics = list(dict.fromkeys(metrics_base + metrics_spike))  # dedup preservando l’ordine

    if not metrics:
        raise ValueError(f"Nessuna metrica trovata. Colonne presenti: {list(df.columns)}")

    # scala temporale + arrotondamento
    scale, xlabel = time_scale_and_label(args.time_unit)
    round_time_decimals = None if args.round_time is not None and args.round_time < 0 else args.round_time

    # genera plot
    for metric in metrics:
        if metric.endswith("util"):
            spaghetti_util(df, metric, scale, xlabel, outdir,
                           round_time_decimals=round_time_decimals, vline_sec=args.vline)
            spaghetti_util_zoom(df, metric, scale, xlabel, outdir,
                                round_time_decimals=round_time_decimals, vline_sec=args.vline)
        else:
            spaghetti_no_legend(df, metric, scale, xlabel, outdir,
                                round_time_decimals=round_time_decimals, vline_sec=args.vline)

    # tabella QoS
    qos_tables(
        df,
        outdir,
        min_legal_completed=args.qos_min_legal_completed,
        max_illegal_completed=args.qos_max_illegal_completed
    )

    print(f"[OK] Grafici e tabelle salvati in: {outdir.resolve()}")

if __name__ == "__main__":
    main()
