#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grafici Analisi del Transitorio: SOLO spaghetti plot per scenario (niente medie).
Salva i PNG in ./transitory/.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from engineering.costants import SEEDS_TRANSITORY
# Colonne attese
TIME_COL = "time"
REP_COL  = "replica"

# Metriche richieste (se mancano nel CSV vengono ignorate)
METRICS = [
    "web_rt_mean", "mit_rt_mean",
    "web_util", "web_throughput",
    "mit_util", "mit_throughput",
    "spikes_count",
]

# ---------- helpers ----------
def guess_available_metrics(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [m for m in candidates if m in df.columns]

def time_scale_and_label(unit: str):
    unit = unit.lower()
    if unit in ("s", "sec", "second", "seconds"):
        return 1.0, "Tempo (s)"
    if unit in ("h", "hour", "hours"):
        return 1/3600.0, "Tempo (h)"
    if unit in ("d", "day", "days"):
        return 1/86400.0, "Tempo (giorni)"
    raise ValueError("time-unit non riconosciuta: usa s|h|day")

def metric_to_label(metric: str) -> str:
    if metric.endswith("rt_mean"):
        return "Tempo di risposta medio (s)"
    if metric.endswith("util"):
        return "Utilizzazione"
    if metric.endswith("throughput"):
        return "Throughput (jobs/s)"
    if metric == "spikes_count":
        return "Spike allocati"
    return metric

def build_label_map(df: pd.DataFrame, seeds: List[int]):
    """
    Ritorna un dict {replica_val -> 'scenario-i (seed)'}.
    Se replica eccede la lista SEEDS, cade su 'scenario-i'.
    """
    label_map = {}
    reps = sorted(df[REP_COL].dropna().unique())
    for r in reps:
        r_int = int(r)
        if 0 <= r_int < len(seeds):
            label_map[r] = f"scenario-{r_int} ({seeds[r_int]})"
        else:
            label_map[r] = f"scenario-{r_int}"
    return label_map

# ---------- plotting ----------
def spaghetti_by_scenario(df: pd.DataFrame,
                          metric: str,
                          scale: float,
                          xlabel: str,
                          outdir: Path,
                          label_map: dict,
                          round_time_decimals: Optional[int] = 0,
                          vline_sec: Optional[float] = None):
    d = df[[TIME_COL, REP_COL, metric]].dropna().copy()
    if round_time_decimals is not None:
        d[TIME_COL] = d[TIME_COL].round(round_time_decimals)
    d = d.sort_values([REP_COL, TIME_COL])

    plt.figure()
    for rep, g in d.groupby(REP_COL):
        gx = g[TIME_COL].to_numpy() * scale
        gy = g[metric].to_numpy()
        label = label_map.get(rep, f"scenario-{int(rep)}")
        plt.plot(gx, gy, linewidth=1.2, label=label)

    if vline_sec is not None:
        plt.axvline(x=vline_sec * scale, linestyle="--")

    plt.xlabel(xlabel)
    plt.ylabel(metric_to_label(metric))
    plt.title(f"{metric} — transitorio (per scenario)")
    plt.grid(True)
    plt.legend(title="Scenario (seed)", loc="best", fontsize="small")
    plt.tight_layout()
    outpath = outdir / f"{metric}_by_scenario.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    return outpath

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Grafici Transitorio: spaghetti plot per scenario")
    ap.add_argument("--csv", type=str, default="results_transitory.csv", help="Percorso CSV checkpoint")
    ap.add_argument("--time-unit", type=str, default="s", choices=["s","h","day"], help="Unità asse X")
    ap.add_argument("--round-time", type=int, default=0, help="Arrotondamento tempo (decimali). -1 per disattivare")
    ap.add_argument("--vline", type=float, default=None, help="Linea verticale a t (in secondi), es. 200000")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir   = Path("transitory")  # <- cartella fissa richiesta
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # metriche disponibili
    metrics = guess_available_metrics(df, METRICS)
    if not metrics:
        raise ValueError(f"Nessuna metrica trovata tra {METRICS}. Colonne: {list(df.columns)}")

    # unità tempo + arrotondamento per allineare checkpoint
    scale, xlabel = time_scale_and_label(args.time_unit)
    round_time_decimals = None if args.round_time is not None and args.round_time < 0 else args.round_time

    # mappa scenario/seed per la legenda
    label_map = build_label_map(df, SEEDS_TRANSITORY)

    # genera SOLO spaghetti plot per scenario
    for metric in metrics:
        spaghetti_by_scenario(df, metric, scale, xlabel, outdir, label_map,
                              round_time_decimals=round_time_decimals, vline_sec=args.vline)

    print(f"[OK] Grafici salvati in: {outdir.resolve()}")

# python plot_transitory.py --csv results_transitory.csv --time-unit day --vline 200000
if __name__ == "__main__":
    main()
