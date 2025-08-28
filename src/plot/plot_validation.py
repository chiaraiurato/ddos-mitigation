import os, sys, csv, math
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

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


def compute_lambda(p: float, l1: float, l2: float) -> float:
    if l1 <= 0 or l2 <= 0:
        return np.nan
    et = p * (1.0 / l1) + (1.0 - p) * (1.0 / l2)
    return 1.0 / et if et > 0 else np.nan

def sort_by_lambda(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for row in rows:
        p = to_float(row.get("ARRIVAL_P"))
        l1 = to_float(row.get("ARRIVAL_L1"))
        l2 = to_float(row.get("ARRIVAL_L2"))
        row["_lambda"] = compute_lambda(p, l1, l2)
    return sorted(rows, key=lambda r: (math.isnan(r["_lambda"]), r["_lambda"]))

def ensure_dir(d: str):
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def plot_series(x, y_mean, xlab, ylab, title, outpath, y_ci=None):
    plt.figure()
    if y_ci is not None and np.isfinite(y_ci).sum() >= 2:
        plt.errorbar(x, y_mean, yerr=y_ci, fmt='o-', capsize=3)
    else:
        plt.plot(x, y_mean, marker='o')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_validation.py validation_results.csv")
        sys.exit(1)

    rows = sort_by_lambda(read_csv(sys.argv[1]))

    lam_theory = np.array([to_float(r["_lambda"]) for r in rows], dtype=float)

    def col(name: str):
        return np.array([to_float(r.get(name)) for r in rows], dtype=float)

    total_time = col("total_time")
    total_arrivals = col("total_arrivals")

    web_util = col("web_util");     web_rt = col("web_rt_mean");     web_thr = col("web_throughput")
    spk_util = col("spikes_util_mean"); spk_rt = col("spikes_rt_mean"); spk_thr = col("spikes_throughput")
    mit_util = col("mit_util");     mit_rt = col("mit_rt_mean");     mit_thr = col("mit_throughput")

    drop_fp = col("drop_fp_rate");  drop_full = col("drop_full_rate")

    web_util_bm_m = col("web_util_bm_mean"); web_util_bm_ci = col("web_util_bm_ci")
    web_rt_bm_m   = col("web_rt_bm_mean");   web_rt_bm_ci   = col("web_rt_bm_ci")
    web_thr_bm_m  = col("web_thr_bm_mean");  web_thr_bm_ci  = col("web_thr_bm_ci")

    spk_util_bm_m = col("spk_util_bm_mean"); spk_util_bm_ci = col("spk_util_bm_ci")
    spk_rt_bm_m   = col("spk_rt_bm_mean");   spk_rt_bm_ci   = col("spk_rt_bm_ci")
    spk_thr_bm_m  = col("spk_thr_bm_mean");  spk_thr_bm_ci  = col("spk_thr_bm_ci")

    mit_util_bm_m = col("mit_util_bm_mean"); mit_util_bm_ci = col("mit_util_bm_ci")
    mit_rt_bm_m   = col("mit_rt_bm_mean");   mit_rt_bm_ci   = col("mit_rt_bm_ci")
    mit_thr_bm_m  = col("mit_thr_bm_mean");  mit_thr_bm_ci  = col("mit_thr_bm_ci")

    def choose(mean_bm, ci_bm, mean_plain):
        if np.isfinite(mean_bm).sum() >= 2:
            return mean_bm, ci_bm
        return mean_plain, None

    with np.errstate(divide='ignore', invalid='ignore'):
        lam_sim = np.where(total_time > 0, total_arrivals / total_time, np.nan)

    X_tot = web_thr + spk_thr
    with np.errstate(divide='ignore', invalid='ignore'):
        quota_web_global = np.where(X_tot > 0, web_thr / X_tot, np.nan)
        efficiency_meas = np.where(lam_sim > 0, X_tot / lam_sim, np.nan)
        efficiency_exp  = np.where(lam_sim > 0, 1.0 - (drop_fp + drop_full) / lam_sim, np.nan)
        mu_hat_web_plain = np.where(web_util > 0, web_thr / web_util, np.nan)

    X_tot_bm_m  = web_thr_bm_m + spk_thr_bm_m
    X_tot_bm_ci = np.where(
        np.isfinite(web_thr_bm_ci) & np.isfinite(spk_thr_bm_ci),
        web_thr_bm_ci + spk_thr_bm_ci,
        np.nan
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        quota_web_bm = np.where(
            (web_thr_bm_m + spk_thr_bm_m) > 0,
            web_thr_bm_m / (web_thr_bm_m + spk_thr_bm_m),
            np.nan
        )
        mu_hat_web_bm = np.where(web_util_bm_m > 0, web_thr_bm_m / web_util_bm_m, np.nan)

    outdir = "validation"
    ensure_dir(outdir)

    y, yci = choose(web_util_bm_m, web_util_bm_ci, web_util)
    plot_series(lam_theory, y, "λ (jobs/s)", "Utilization", "Web Utilization vs λ",
                os.path.join(outdir, "web_util_vs_lambda.png"), y_ci=yci)

    y, yci = choose(spk_util_bm_m, spk_util_bm_ci, spk_util)
    plot_series(lam_theory, y, "λ (jobs/s)", "Utilization", "Spike Avg Utilization vs λ",
                os.path.join(outdir, "spike_util_vs_lambda.png"), y_ci=yci)

    y, yci = choose(mit_util_bm_m, mit_util_bm_ci, mit_util)
    plot_series(lam_theory, y, "λ (jobs/s)", "Utilization", "Mitigation Utilization vs λ",
                os.path.join(outdir, "mitigation_util_vs_lambda.png"), y_ci=yci)

    y, yci = choose(web_rt_bm_m, web_rt_bm_ci, web_rt)
    plot_series(lam_theory, y, "λ (jobs/s)", "Response Time (s)", "Web Response Time vs λ",
                os.path.join(outdir, "web_rt_vs_lambda.png"), y_ci=yci)

    y, yci = choose(spk_rt_bm_m, spk_rt_bm_ci, spk_rt)
    plot_series(lam_theory, y, "λ (jobs/s)", "Response Time (s)", "Spike Response Time vs λ",
                os.path.join(outdir, "spike_rt_vs_lambda.png"), y_ci=yci)

    y, yci = choose(mit_rt_bm_m, mit_rt_bm_ci, mit_rt)
    plot_series(lam_theory, y, "λ (jobs/s)", "Response Time (s)", "Mitigation Response Time vs λ",
                os.path.join(outdir, "mitigation_rt_vs_lambda.png"), y_ci=yci)

    y, yci = choose(web_thr_bm_m, web_thr_bm_ci, web_thr)
    plot_series(lam_theory, y, "λ (jobs/s)", "Throughput (jobs/s)", "Web Throughput vs λ",
                os.path.join(outdir, "web_thr_vs_lambda.png"), y_ci=yci)

    y, yci = choose(spk_thr_bm_m, spk_thr_bm_ci, spk_thr)
    plot_series(lam_theory, y, "λ (jobs/s)", "Throughput (jobs/s)", "Spike Throughput vs λ",
                os.path.join(outdir, "spike_thr_vs_lambda.png"), y_ci=yci)

    y, yci = choose(mit_thr_bm_m, mit_thr_bm_ci, mit_thr)
    plot_series(lam_theory, y, "λ (jobs/s)", "Throughput (jobs/s)", "Mitigation Throughput vs λ",
                os.path.join(outdir, "mitigation_thr_vs_lambda.png"), y_ci=yci)

    if np.isfinite(X_tot_bm_m).sum() >= 2:
        plot_series(lam_theory, X_tot_bm_m, "λ (jobs/s)", "Throughput totale (jobs/s)",
                    "Total Throughput vs λ",
                    os.path.join(outdir, "total_throughput_vs_lambda.png"),
                    y_ci=X_tot_bm_ci)
    else:
        plot_series(lam_theory, X_tot, "λ (jobs/s)", "Throughput totale (jobs/s)",
                    "Total Throughput vs λ",
                    os.path.join(outdir, "total_throughput_vs_lambda.png"))

    print("\nλ_theory\tλ_sim\tX_web\tX_spike\tX_tot\tX_tot_bm\teta_meas\teta_exp")
    for i in range(len(lam_theory)):
        def f(z): return f"{z:.6f}" if np.isfinite(z) else "nan"
        print(f"{f(lam_theory[i])}\t{f(lam_sim[i])}\t{f(web_thr[i])}\t{f(spk_thr[i])}\t"
              f"{f(X_tot[i])}\t{f(X_tot_bm_m[i])}\t{f(efficiency_meas[i])}\t"
              f"{f(efficiency_exp[i])}")

    print(f"\nGrafici generati nella cartella: {outdir}")


#  python3 plots_validation.py results_standard.csv
if __name__ == "__main__":
    main()