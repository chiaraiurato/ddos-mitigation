import os, sys, csv
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def read_csv(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        raise RuntimeError("CSV vuoto.")
    return rows

def to_float(x: Any, default=np.nan) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default

def ensure_dir(d: str):
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def plot_series_cat(labels, y, xlab, ylab, title, outpath):
    x = np.arange(len(labels), dtype=float)
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xticks(x, labels)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_series_cat_int_y(labels, y, xlab, ylab, title, outpath):
    x = np.arange(len(labels), dtype=float)
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xticks(x, labels)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def choose_prefer_bm(mean_bm: np.ndarray, mean_point: np.ndarray) -> np.ndarray:
    return mean_bm if np.isfinite(mean_bm).sum() >= 2 else mean_point

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_validation_improved.py results_validation_ml_analysis.csv")
        sys.exit(1)

    rows = read_csv(sys.argv[1])

    def scen_key(s: str) -> float:
        s = (s or "").strip().lower()
        if s.startswith("x") and s[1:].isdigit():
            return float(s[1:])
        return float("inf")

    rows.sort(key=lambda r: scen_key(r.get("scenario", "")))

    scenarios = [r.get("scenario", "") for r in rows]

    def col(name: str):
        return np.array([to_float(r.get(name)) for r in rows], dtype=float)

    total_time = col("total_time")
    total_arrivals = col("total_arrivals")
    spikes_count = col("spikes_count")

    web_util = col("web_util");           web_rt = col("web_rt_mean");           web_thr = col("web_throughput")
    spk0_util = col("spike0_util");       spk0_rt = col("spike0_rt_mean");       spk0_thr = col("spike0_throughput")
    mit_util = col("mit_util");           mit_rt = col("mit_rt_mean");           mit_thr = col("mit_throughput")

    ana_util_point = col("analysis_util")
    ana_rt_point   = col("analysis_rt_mean")
    ana_thr_point  = col("analysis_throughput")

    drop_fp = col("drop_fp_rate");  drop_full = col("drop_full_rate")

    web_util_bm_m = col("web_util_bm_mean")
    web_rt_bm_m   = col("web_rt_bm_mean")
    web_thr_bm_m  = col("web_thr_bm_mean")

    spk0_util_bm_m = col("spike0_util_bm_mean")
    spk0_rt_bm_m   = col("spike0_rt_bm_mean")
    spk0_thr_bm_m  = col("spike0_thr_bm_mean")

    mit_util_bm_m = col("mit_util_bm_mean")
    mit_rt_bm_m   = col("mit_rt_bm_mean")
    mit_thr_bm_m  = col("mit_thr_bm_mean")

    ana_util_bm_m = col("ana_util_bm_mean")
    ana_thr_bm_m  = col("ana_thr_bm_mean")

    web_util_used = choose_prefer_bm(web_util_bm_m, web_util)
    web_rt_used   = choose_prefer_bm(web_rt_bm_m,   web_rt)
    web_thr_used  = choose_prefer_bm(web_thr_bm_m,  web_thr)

    spk0_util_used = choose_prefer_bm(spk0_util_bm_m, spk0_util)
    spk0_rt_used   = choose_prefer_bm(spk0_rt_bm_m,   spk0_rt)
    spk0_thr_used  = choose_prefer_bm(spk0_thr_bm_m,  spk0_thr)

    mit_util_used = choose_prefer_bm(mit_util_bm_m, mit_util)
    mit_rt_used   = choose_prefer_bm(mit_rt_bm_m,   mit_rt)
    mit_thr_used  = choose_prefer_bm(mit_thr_bm_m,  mit_thr)

    ana_util_used = choose_prefer_bm(ana_util_bm_m, ana_util_point)
    ana_thr_used  = choose_prefer_bm(ana_thr_bm_m,  ana_thr_point)
    ana_rt_used   = ana_rt_point  

    with np.errstate(divide='ignore', invalid='ignore'):
        lam_sim = np.where(total_time > 0, total_arrivals / total_time, np.nan)
        X_tot_web_spk0 = web_thr_used + spk0_thr_used
        efficiency_meas = np.where(lam_sim > 0, X_tot_web_spk0 / lam_sim, np.nan)
        efficiency_exp  = np.where(lam_sim > 0, 1.0 - (drop_fp + drop_full) / lam_sim, np.nan)

    outdir = "validation_improved"
    ensure_dir(outdir)

    plot_series_cat(scenarios, web_util_used,  "Scenario", "Utilization",
                    "Web Utilization vs Scenario",
                    os.path.join(outdir, "web_util_vs_scenario.png"))

    plot_series_cat(scenarios, spk0_util_used, "Scenario", "Utilization",
                    "Spike-0 Utilization vs Scenario",
                    os.path.join(outdir, "spike0_util_vs_scenario.png"))

    plot_series_cat(scenarios, mit_util_used,  "Scenario", "Utilization",
                    "Mitigation Utilization vs Scenario",
                    os.path.join(outdir, "mitigation_util_vs_scenario.png"))

    plot_series_cat(scenarios, ana_util_used,  "Scenario", "Utilization",
                    "Analysis ML Utilization vs Scenario",
                    os.path.join(outdir, "analysis_util_vs_scenario.png"))

    plot_series_cat(scenarios, web_rt_used, "Scenario", "Response Time (s)",
                    "Web Response Time vs Scenario",
                    os.path.join(outdir, "web_rt_vs_scenario.png"))

    plot_series_cat(scenarios, spk0_rt_used, "Scenario", "Response Time (s)",
                    "Spike-0 Response Time vs Scenario",
                    os.path.join(outdir, "spike0_rt_vs_scenario.png"))

    plot_series_cat(scenarios, mit_rt_used, "Scenario", "Response Time (s)",
                    "Mitigation Response Time vs Scenario",
                    os.path.join(outdir, "mitigation_rt_vs_scenario.png"))

    plot_series_cat(scenarios, ana_rt_used, "Scenario", "Response Time (s)",
                    "Analysis ML Response Time vs Scenario",
                    os.path.join(outdir, "analysis_rt_vs_scenario.png"))

    plot_series_cat(scenarios, web_thr_used, "Scenario", "Throughput (jobs/s)",
                    "Web Throughput vs Scenario",
                    os.path.join(outdir, "web_thr_vs_scenario.png"))

    plot_series_cat(scenarios, spk0_thr_used, "Scenario", "Throughput (jobs/s)",
                    "Spike-0 Throughput vs Scenario",
                    os.path.join(outdir, "spike0_thr_vs_scenario.png"))

    plot_series_cat(scenarios, mit_thr_used, "Scenario", "Throughput (jobs/s)",
                    "Mitigation Throughput vs Scenario",
                    os.path.join(outdir, "mitigation_thr_vs_scenario.png"))

    plot_series_cat(scenarios, ana_thr_used, "Scenario", "Throughput (jobs/s)",
        "Analysis ML Throughput vs Scenario",
        os.path.join(outdir, "analysis_thr_vs_scenario.png"))

    plot_series_cat(scenarios, X_tot_web_spk0, "Scenario", "Throughput (jobs/s)",
                    "Web+Spike-0 Throughput vs Scenario",
                    os.path.join(outdir, "web_spike0_total_throughput_vs_scenario.png"))

    plot_series_cat_int_y(
        scenarios, spikes_count, "Scenario", "# Spike allocati",
        "Spike allocati (linea) vs Scenario",
        os.path.join(outdir, "spikes_count_line_vs_scenario.png")
    )

    print("\nScenario\tspikes\tλ_sim\tX_web\tX_spk0\tX_web+spk0\tη_meas\tη_exp")
    for i in range(len(scenarios)):
        def f(z): return f"{z:.6f}" if np.isfinite(z) else "nan"
        print(f"{scenarios[i]}\t{int(spikes_count[i]) if np.isfinite(spikes_count[i]) else 'nan'}\t"
              f"{f(lam_sim[i])}\t{f(web_thr_used[i])}\t{f(spk0_thr_used[i])}\t"
              f"{f(X_tot_web_spk0[i])}\t{f(efficiency_meas[i])}\t{f(efficiency_exp[i])}")

    print(f"\nGrafici generati nella cartella: {outdir}")


# python3 plot_validation_improved.py results_validation_ml_analysis.csv
if __name__ == "__main__":
    main()
