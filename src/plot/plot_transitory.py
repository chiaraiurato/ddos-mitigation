import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

SEEDS = [123456789, 653476254, 734861870, 976578247, 364519872, 984307865, 546274352]

TIME_COL = "time"
REP_COL  = "replica"

METRICS = [
    "web_rt_mean", "web_util", "web_throughput",
    "spike0_rt_mean", "spike0_util", "spike0_throughput",
    "mit_rt_mean", "mit_util", "mit_throughput",
    "spikes_count",
]

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

def pretty_title(metric: str) -> str:
    if metric.startswith("web_"):
        who = "Web"
    elif metric.startswith("spike0_"):
        who = "Spike-0"
    elif metric.startswith("mit_"):
        who = "Mitigation"
    else:
        who = metric.split("_")[0].capitalize()
    if metric.endswith("rt_mean"):
        what = "RT medio"
    elif metric.endswith("util"):
        what = "Utilizzazione"
    elif metric.endswith("throughput"):
        what = "Throughput"
    else:
        what = metric
    return f"{who} — {what} (transitorio, per scenario)"

def build_label_map(df: pd.DataFrame, seeds: List[int]):
    """Ritorna un dict {replica_val -> 'scenario-i (seed)'}. """
    label_map = {}
    reps = sorted(df[REP_COL].dropna().unique())
    for r in reps:
        r_int = int(r)
        if 0 <= r_int < len(seeds):
            label_map[r] = f"scenario-{r_int} ({seeds[r_int]})"
        else:
            label_map[r] = f"scenario-{r_int}"
    return label_map

def spaghetti_by_scenario(df: pd.DataFrame,
                          metric: str,
                          scale: float,
                          xlabel: str,
                          outdir: Path,
                          label_map: dict,
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
    for rep, g in d.groupby(REP_COL):
        gx = g[TIME_COL].to_numpy() * scale
        gy = g[metric].to_numpy()
        label = label_map.get(rep, f"scenario-{int(rep)}")
        plt.plot(gx, gy, linewidth=1.2, label=label)

    if vline_sec is not None:
        plt.axvline(x=vline_sec * scale, linestyle="--")

    if metric.endswith("util"):
        headroom = 1e-3
        ymax_obs = float(d[metric].max())
        ax.set_ylim(0.0, max(1.0 + headroom, ymax_obs + headroom))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.axhline(1.0, color='gray', linewidth=0.8, alpha=0.6)

    plt.xlabel(xlabel)
    plt.ylabel(metric_to_label(metric))
    plt.title(pretty_title(metric))
    plt.grid(True)
    plt.legend(title="Scenario (seed)", loc="best", fontsize="small")
    plt.tight_layout()
    outpath = outdir / f"{metric}_by_scenario.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    return outpath

def spaghetti_util_zoom_by_scenario(df: pd.DataFrame,
                                    metric: str,
                                    scale: float,
                                    xlabel: str,
                                    outdir: Path,
                                    label_map: dict,
                                    round_time_decimals: Optional[int] = 0,
                                    vline_sec: Optional[float] = None):

    d = df[[TIME_COL, REP_COL, metric]].dropna().copy()
    if d.empty:
        return None

    if round_time_decimals is not None:
        d[TIME_COL] = d[TIME_COL].round(round_time_decimals)
    d = d.sort_values([REP_COL, TIME_COL])

    ymin_obs = float(d[metric].min())
    ymax_obs = float(d[metric].max())

    spread = max(ymax_obs - ymin_obs, 1e-6)
    delta  = max(3e-4, spread * 0.30)
    low  = max(0.0, ymin_obs - delta)
    high = max(1.0 + 1e-3, ymax_obs + delta)

    plt.figure()
    ax = plt.gca()
    for rep, g in d.groupby(REP_COL):
        gx = g[TIME_COL].to_numpy() * scale
        gy = g[metric].to_numpy()
        label = label_map.get(rep, f"scenario-{int(rep)}")
        plt.plot(gx, gy, linewidth=1.2, label=label)

    if vline_sec is not None:
        plt.axvline(x=vline_sec * scale, linestyle="--")

    ax.set_ylim(low, high)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.axhline(1.0, color='gray', linewidth=0.8, alpha=0.6)

    plt.xlabel(xlabel)
    plt.ylabel(metric_to_label(metric))
    plt.title(pretty_title(metric) + " (zoom)")
    plt.grid(True)
    plt.legend(title="Scenario (seed)", loc="best", fontsize="small")
    plt.tight_layout()
    outpath = outdir / f"{metric}_by_scenario_zoom.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    return outpath

def main():
    ap = argparse.ArgumentParser(description="Grafici Transitorio: spaghetti plot per scenario")
    ap.add_argument("--csv", type=str, default="results_transitory.csv", help="Percorso CSV checkpoint")
    ap.add_argument("--time-unit", type=str, default="s", choices=["s","h","day"], help="Unità asse X")
    ap.add_argument("--round-time", type=int, default=0, help="Arrotondamento tempo (decimali). -1 per disattivare")
    ap.add_argument("--vline", type=float, default=None, help="Linea verticale a t (in secondi), es. 200000")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir   = Path("transitory")
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    metrics = guess_available_metrics(df, METRICS)
    if not metrics:
        raise ValueError(f"Nessuna metrica trovata tra {METRICS}. Colonne: {list(df.columns)}")

    scale, xlabel = time_scale_and_label(args.time_unit)
    round_time_decimals = None if args.round_time is not None and args.round_time < 0 else args.round_time

    label_map = build_label_map(df, SEEDS)

    for metric in metrics:
        spaghetti_by_scenario(df, metric, scale, xlabel, outdir, label_map,
                              round_time_decimals=round_time_decimals, vline_sec=args.vline)
        if metric.endswith("util"):
            spaghetti_util_zoom_by_scenario(df, metric, scale, xlabel, outdir, label_map,
                                            round_time_decimals=round_time_decimals, vline_sec=args.vline)

    print(f"[OK] Grafici salvati in: {outdir.resolve()}")

# python3 plot_transitory.py --csv results_transitory_baseline.csv --time-unit s 
if __name__ == "__main__":
    main()
