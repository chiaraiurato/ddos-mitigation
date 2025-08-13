# validation_run.py
import csv
import importlib
import numpy as np
import simpy
import math

import engineering.distributions as _dist_mod
import engineering.costants as _const_mod

from controller.verification_run import DDoSSystem
from engineering.statistics import batch_means, window_util_thr


# Scenari di default se non presenti in engineering.costants.VALIDATION_SCENARIOS
# Scenari di default se non presenti in engineering.costants.VALIDATION_SCENARIOS
_DEFAULT_VALIDATION_SCENARIOS = [
    ("x1",      0.03033,    0.4044,     12.9289),
    ("x2",      0.03033,    0.8088,     35.9778),
    ("x5",      0.03033,    2.022,      64.6445),
    ("x10",     0.03033,    4.044,      129.289),
    ("x40",     0.03033,    16.176,     517.156),
]


def _fmt(x):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "nan"
        return f"{x:.6f}"
    except Exception:
        return str(x)


def _bm_or_nan(samples, batch_size):
    """Applica batch_means con gestione campioni insufficienti → (mean, ci) oppure (nan, nan)."""
    try:
        if samples is None:
            return (float("nan"), float("nan"))
        n = len(samples)
        if n < batch_size or (n // batch_size) < 2:
            return (float("nan"), float("nan"))
        m, ci = batch_means(samples, batch_size)
        return (float(m), float(ci))
    except Exception:
        return (float("nan"), float("nan"))


def _bm_windows_or_nan(samples_windows, windows_per_batch):
    """Batch-means su sequenza di campioni per finestre; richiede >= 2 batch."""
    try:
        if samples_windows is None:
            return (float("nan"), float("nan"))
        n = len(samples_windows)
        if n < windows_per_batch or (n // windows_per_batch) < 2:
            return (float("nan"), float("nan"))
        m, ci = batch_means(samples_windows, windows_per_batch)
        return (float(m), float(ci))
    except Exception:
        return (float("nan"), float("nan"))


def _merge_spikes_completion_times_and_rts(spikes):
    """
    Merge-stable dei completamenti degli Spike per ottenere una sequenza globale
    (ordinata per tempo di completamento) di response times, così da poter fare
    batch means sui completamenti aggregati.
    """
    # Puntatori per ogni spike
    idxs = [0] * len(spikes)
    merged_times = []
    merged_rts = []

    while True:
        min_t = math.inf
        min_i = -1
        for i, s in enumerate(spikes):
            if idxs[i] < len(s.completion_times):
                t = s.completion_times[idxs[i]]
                if t < min_t:
                    min_t = t
                    min_i = i
        if min_i < 0:
            break
        merged_times.append(min_t)
        merged_rts.append(spikes[min_i].completed_jobs[idxs[min_i]])
        idxs[min_i] += 1

    return merged_times, merged_rts


def run_validation():
    """
    Esegue più run variando (ARRIVAL_P, ARRIVAL_L1, ARRIVAL_L2) dell'iperesponenziale
    e salva i risultati in 'validation_results.csv', includendo media±IC (95%) via Batch Means
    per utilization/throughput (a finestre) e response time (a completamenti) di Web/Spike agg./Mitigation.
    """
    scenarios = getattr(_const_mod, "VALIDATION_SCENARIOS", _DEFAULT_VALIDATION_SCENARIOS)

    # Parametri BM (fallback a quelli generali se non definiti specifici per validation)
    BATCH_SIZE_RT = getattr(_const_mod, "BATCH_SIZE_VALIDATION", getattr(_const_mod, "BATCH_SIZE", 1024))
    TIME_WINDOW = getattr(_const_mod, "TIME_WINDOW_VALIDATION", getattr(_const_mod, "TIME_WINDOW", 30.0))
    WINDOWS_PER_BATCH = getattr(_const_mod, "TIME_WINDOWS_PER_BATCH_VALIDATION",
                                getattr(_const_mod, "TIME_WINDOWS_PER_BATCH", 32))

    # salva originali per ripristino
    orig_P  = getattr(_const_mod, "ARRIVAL_P",  None)
    orig_L1 = getattr(_const_mod, "ARRIVAL_L1", None)
    orig_L2 = getattr(_const_mod, "ARRIVAL_L2", None)

    out_path = "validation_results.csv"
    fieldnames = [
        # scenari
        "scenario", "ARRIVAL_P", "ARRIVAL_L1", "ARRIVAL_L2",
        # run base
        "total_time", "total_arrivals",
        # metriche base (come prima)
        "web_util", "web_rt_mean", "web_throughput",
        "spikes_count", "spikes_util_mean", "spikes_rt_mean", "spikes_throughput",
        "mit_util", "mit_rt_mean", "mit_throughput",
        "drop_fp_rate", "drop_full_rate",
        # ====== NUOVO: Batch Means (mean ± CI) ======
        # Web
        "web_util_bm_mean", "web_util_bm_ci",
        "web_thr_bm_mean",  "web_thr_bm_ci",
        "web_rt_bm_mean",   "web_rt_bm_ci",
        # Spikes (aggregati)
        "spk_util_bm_mean", "spk_util_bm_ci",
        "spk_thr_bm_mean",  "spk_thr_bm_ci",
        "spk_rt_bm_mean",   "spk_rt_bm_ci",
        # Mitigation
        "mit_util_bm_mean", "mit_util_bm_ci",
        "mit_thr_bm_mean",  "mit_thr_bm_ci",
        "mit_rt_bm_mean",   "mit_rt_bm_ci",
        # info BM utili per debug/relazione
        "bm_rt_batch_size", "bm_win_size", "bm_windows_per_batch"
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name, Pval, L1val, L2val in scenarios:
            # set parametri iperesponenziale
            _const_mod.ARRIVAL_P = Pval
            _const_mod.ARRIVAL_L1 = L1val
            _const_mod.ARRIVAL_L2 = L2val
            importlib.reload(_dist_mod)   # assicura che get_interarrival_time veda i nuovi valori

            # run singola in modalità 'validation' (iperesponenziale come 'standard')
            env = simpy.Environment()
            system = DDoSSystem(env, "validation")
            env.run()
            now = env.now

            web = system.web_server
            spikes = system.spike_servers
            center = system.mitigation_manager.center

            # ===== metriche base (come prima) =====
            web_util = web.busy_time / now if now > 0 else 0.0
            web_rt_mean = (np.mean(web.completed_jobs) if web.completed_jobs else float("nan"))
            web_thr = web.total_completions / now if now > 0 else 0.0

            spikes_count = len(spikes)
            spikes_utils = [(s.busy_time / now) for s in spikes] if spikes else []
            spikes_util_mean = float(np.mean(spikes_utils)) if spikes_utils else 0.0
            spikes_rts = np.concatenate([np.array(s.completed_jobs) for s in spikes]) if spikes else np.array([])
            spikes_rt_mean = float(np.mean(spikes_rts)) if spikes_rts.size > 0 else float("nan")
            spikes_thr = (sum(s.total_completions for s in spikes) / now) if (spikes and now > 0) else 0.0

            mit_util = center.busy_time / now if now > 0 else 0.0
            mit_rt_mean = (np.mean(center.completed_jobs) if center.completed_jobs else float("nan"))
            mit_thr = center.total_completions / now if now > 0 else 0.0

            drop_fp_rate = system.metrics.get("false_positives", 0) / now if now > 0 else 0.0
            drop_full_rate = system.metrics.get("discarded_mitigation", 0) / now if now > 0 else 0.0

            # ===== Batch Means: WEB =====
            web_util_samples, web_thr_samples = window_util_thr(
                getattr(web, "busy_periods", []), web.completion_times, TIME_WINDOW, now
            )
            web_util_bm_mean, web_util_bm_ci = _bm_windows_or_nan(web_util_samples, WINDOWS_PER_BATCH)
            web_thr_bm_mean,  web_thr_bm_ci  = _bm_windows_or_nan(web_thr_samples,  WINDOWS_PER_BATCH)
            web_rt_bm_mean,   web_rt_bm_ci   = _bm_or_nan(web.completed_jobs, BATCH_SIZE_RT)

            # ===== Batch Means: SPIKES (aggregati) =====
            # util e thr per finestra: media util su spike; somma throughput su spike
            spk_util_matrix = []
            spk_thr_matrix = []
            for s in spikes:
                u_s, th_s = window_util_thr(getattr(s, "busy_periods", []), s.completion_times, TIME_WINDOW, now)
                spk_util_matrix.append(np.array(u_s, dtype=float))
                spk_thr_matrix.append(np.array(th_s, dtype=float))

            if spk_util_matrix:
                # allinea per lunghezza minima (in teoria sono uguali, ma per sicurezza)
                min_len_u = min(len(x) for x in spk_util_matrix)
                min_len_t = min(len(x) for x in spk_thr_matrix)
                U = np.vstack([x[:min_len_u] for x in spk_util_matrix])   # shape: (#spike, #win)
                T = np.vstack([x[:min_len_t] for x in spk_thr_matrix])
                spk_util_windows = np.mean(U, axis=0)                     # media util per finestra
                spk_thr_windows  = np.sum(T, axis=0)                      # somma thr per finestra
            else:
                spk_util_windows = []
                spk_thr_windows = []

            spk_util_bm_mean, spk_util_bm_ci = _bm_windows_or_nan(list(spk_util_windows), WINDOWS_PER_BATCH)
            spk_thr_bm_mean,  spk_thr_bm_ci  = _bm_windows_or_nan(list(spk_thr_windows),  WINDOWS_PER_BATCH)

            # Response time aggregato spike: merge per ordine di completamento
            _, spk_rts_merged = _merge_spikes_completion_times_and_rts(spikes)
            spk_rt_bm_mean, spk_rt_bm_ci = _bm_or_nan(spk_rts_merged, BATCH_SIZE_RT)

            # ===== Batch Means: MITIGATION =====
            mit_util_samples, mit_thr_samples = window_util_thr(
                center.busy_periods, center.completion_times, TIME_WINDOW, now
            )
            mit_util_bm_mean, mit_util_bm_ci = _bm_windows_or_nan(mit_util_samples, WINDOWS_PER_BATCH)
            mit_thr_bm_mean,  mit_thr_bm_ci  = _bm_windows_or_nan(mit_thr_samples,  WINDOWS_PER_BATCH)
            mit_rt_bm_mean,   mit_rt_bm_ci   = _bm_or_nan(center.completed_jobs, BATCH_SIZE_RT)

            # ===== Scrittura CSV =====
            writer.writerow({
                "scenario": name,
                "ARRIVAL_P": Pval, "ARRIVAL_L1": L1val, "ARRIVAL_L2": L2val,
                "total_time": now, "total_arrivals": system.metrics["total_arrivals"],
                "web_util": web_util, "web_rt_mean": web_rt_mean, "web_throughput": web_thr,
                "spikes_count": spikes_count, "spikes_util_mean": spikes_util_mean,
                "spikes_rt_mean": spikes_rt_mean, "spikes_throughput": spikes_thr,
                "mit_util": mit_util, "mit_rt_mean": mit_rt_mean, "mit_throughput": mit_thr,
                "drop_fp_rate": drop_fp_rate, "drop_full_rate": drop_full_rate,
                # BM Web
                "web_util_bm_mean": web_util_bm_mean, "web_util_bm_ci": web_util_bm_ci,
                "web_thr_bm_mean":  web_thr_bm_mean,  "web_thr_bm_ci":  web_thr_bm_ci,
                "web_rt_bm_mean":   web_rt_bm_mean,   "web_rt_bm_ci":   web_rt_bm_ci,
                # BM Spikes (aggregati)
                "spk_util_bm_mean": spk_util_bm_mean, "spk_util_bm_ci": spk_util_bm_ci,
                "spk_thr_bm_mean":  spk_thr_bm_mean,  "spk_thr_bm_ci":  spk_thr_bm_ci,
                "spk_rt_bm_mean":   spk_rt_bm_mean,   "spk_rt_bm_ci":   spk_rt_bm_ci,
                # BM Mitigation
                "mit_util_bm_mean": mit_util_bm_mean, "mit_util_bm_ci": mit_util_bm_ci,
                "mit_thr_bm_mean":  mit_thr_bm_mean,  "mit_thr_bm_ci":  mit_thr_bm_ci,
                "mit_rt_bm_mean":   mit_rt_bm_mean,   "mit_rt_bm_ci":   mit_rt_bm_ci,
                # info BM
                "bm_rt_batch_size": BATCH_SIZE_RT,
                "bm_win_size": TIME_WINDOW,
                "bm_windows_per_batch": WINDOWS_PER_BATCH,
            })

            print(f"[validation+BM] scenario={name}: "
                  f"web_rt_bm={_fmt(web_rt_bm_mean)}±{_fmt(web_rt_bm_ci)}, "
                  f"spk_rt_bm={_fmt(spk_rt_bm_mean)}±{_fmt(spk_rt_bm_ci)}, "
                  f"mit_rt_bm={_fmt(mit_rt_bm_mean)}±{_fmt(mit_rt_bm_ci)} | "
                  f"web_util_bm={_fmt(web_util_bm_mean)}±{_fmt(web_util_bm_ci)}")

    # ripristina i parametri originali e ricarica distributions
    _const_mod.ARRIVAL_P  = orig_P
    _const_mod.ARRIVAL_L1 = orig_L1
    _const_mod.ARRIVAL_L2 = orig_L2
    importlib.reload(_dist_mod)

    print(f"\n[validation] risultati (con Batch Means) salvati in: {out_path}")
