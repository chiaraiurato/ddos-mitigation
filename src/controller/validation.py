import csv
import importlib
import numpy as np
import simpy

import engineering.distributions as _dist_mod
import engineering.costants as _const_mod

from controller.verification_run import DDoSSystem  # riuso la stessa classe
from engineering.costants import *       # per N_ARRIVALS, ecc.


# Scenari di default se non presenti in engineering.costants.VALIDATION_SCENARIOS
_DEFAULT_VALIDATION_SCENARIOS = [
    ("paper", 0.03033,  1.2132,   38.7867),
    ("x5",    0.03033,  2.022,    64.6445),
    ("x10",   0.03033,  4.044,    129.289),
    ("x40",   0.03033, 16.176,    517.156),
]


def _fmt(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        return f"{x:.6f}"
    except Exception:
        return str(x)


def run_validation():
    """
    Esegue più run variando (ARRIVAL_P, ARRIVAL_L1, ARRIVAL_L2) dell'iperesponenziale
    e salva i risultati in 'validation_results.csv'.
    Gli scenari si leggono da engineering.costants.VALIDATION_SCENARIOS se presente,
    altrimenti usa 4 scenari di default (paper, x5, x10, x40).
    """
    scenarios = getattr(_const_mod, "VALIDATION_SCENARIOS", _DEFAULT_VALIDATION_SCENARIOS)

    # salva originali per ripristino
    orig_P  = getattr(_const_mod, "ARRIVAL_P",  None)
    orig_L1 = getattr(_const_mod, "ARRIVAL_L1", None)
    orig_L2 = getattr(_const_mod, "ARRIVAL_L2", None)

    out_path = "validation_results.csv"
    fieldnames = [
        "scenario", "ARRIVAL_P", "ARRIVAL_L1", "ARRIVAL_L2",
        "total_time", "total_arrivals",
        "web_util", "web_rt_mean", "web_throughput",
        "spikes_count", "spikes_util_mean", "spikes_rt_mean", "spikes_throughput",
        "mit_util", "mit_rt_mean", "mit_throughput",
        "drop_fp_rate", "drop_full_rate"
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name, Pval, L1val, L2val in scenarios:
            # setta i parametri dell'iperesponenziale negli stessi constants usati da distributions
            _const_mod.ARRIVAL_P = Pval
            _const_mod.ARRIVAL_L1 = L1val
            _const_mod.ARRIVAL_L2 = L2val
            importlib.reload(_dist_mod)   # assicura che get_interarrival_time veda i nuovi valori

            # run singola in modalità 'validation' (iperesponenziale come standard)
            env = simpy.Environment()
            system = DDoSSystem(env, "validation")
            env.run()
            now = env.now

            web = system.web_server
            spikes = system.spike_servers
            center = system.mitigation_manager.center

            # metriche
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

            writer.writerow({
                "scenario": name,
                "ARRIVAL_P": Pval, "ARRIVAL_L1": L1val, "ARRIVAL_L2": L2val,
                "total_time": now, "total_arrivals": system.metrics["total_arrivals"],
                "web_util": web_util, "web_rt_mean": web_rt_mean, "web_throughput": web_thr,
                "spikes_count": spikes_count, "spikes_util_mean": spikes_util_mean,
                "spikes_rt_mean": spikes_rt_mean, "spikes_throughput": spikes_thr,
                "mit_util": mit_util, "mit_rt_mean": mit_rt_mean, "mit_throughput": mit_thr,
                "drop_fp_rate": drop_fp_rate, "drop_full_rate": drop_full_rate
            })

            print(f"[validation] scenario={name}: "f"P={Pval}, L1={L1val}, L2={L2val} | "
                  f"web_util={_fmt(web_util)}, web_rt={_fmt(web_rt_mean)}, "
                  f"spikes={spikes_count}, mit_util={_fmt(mit_util)}")

    # ripristina i parametri originali (se c'erano) e ricarica distributions
    _const_mod.ARRIVAL_P  = orig_P
    _const_mod.ARRIVAL_L1 = orig_L1
    _const_mod.ARRIVAL_L2 = orig_L2
    importlib.reload(_dist_mod)

    print(f"\n[validation] risultati salvati in: {out_path}")
