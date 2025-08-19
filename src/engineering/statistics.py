import numpy as np
from library.rvms import idfStudent
from engineering.costants import BATCH_SIZE, N_BATCH, CONFIDENCE_LEVEL
import os
import math
import pandas as pd

def _compute_batch_means(data, batch_size, burn_in=0, k_max=None, estimate_ci=False):
    """
    Calcola il metodo del batch means per i dati forniti.
    """
    x = np.asarray(list(data), dtype=float)
    if burn_in > 0 and x.size > burn_in:
        x = x[burn_in:]
    n = x.size
    if n < batch_size:
        raise ValueError(f"Not enough samples after burn-in: n={n} < B={batch_size}")
    k = n // batch_size
    if k_max is not None:
        k = min(k, int(k_max))
    if estimate_ci and k < 2:
        raise ValueError(f"Need at least 2 batches, got k={k} (n={n}, B={batch_size}).")
    trimmed = x[:k * batch_size]
    means = trimmed.reshape(k, batch_size).mean(axis=1)
    return means, k

def make_batch_means_series(data, b, burn_in=0, k_max=None):
    """
    Crea una serie temporale di batch means dai dati forniti.
    """
    means, _ = _compute_batch_means(data, b, burn_in=burn_in, k_max=k_max, estimate_ci=False)
    return means

def batch_means(data, batch_size=None, n_batches=None, confidence=None, burn_in=0):
    """
    Calcola l'intervallo di confidenza per il batch means.
    """
    if batch_size is None: batch_size = BATCH_SIZE
    if n_batches  is None: n_batches  = N_BATCH
    if confidence is None: confidence = CONFIDENCE_LEVEL
    means, k = _compute_batch_means(data, batch_size, burn_in=burn_in, k_max=n_batches, estimate_ci=True)
    xbar = float(np.mean(means))
    s = float(np.std(means, ddof=1))
    alpha = 1.0 - float(confidence); u = 1.0 - alpha / 2.0
    t_star = float(idfStudent(k - 1, u))
    half_width = t_star * (s / np.sqrt(k))
    return xbar, half_width

def window_util_thr(busy_periods, completion_times, window, now):
    if window <= 0 or now <= 0:
        return [], []
    nwin = int(now // window)
    if nwin <= 0:
        return [], []

    util_samples, thr_samples = [], []
    comp = sorted(completion_times)
    ncomp, idx = len(comp), 0

    for k in range(nwin):
        a, b = k * window, (k + 1) * window

        busy_in_window = 0.0
        for (s, e) in busy_periods:
            if e <= a or s >= b:
                continue
            busy_in_window += max(0.0, min(e, b) - max(s, a))
        util_samples.append(busy_in_window / window)

        count = 0
        while idx < ncomp and comp[idx] < b:
            if comp[idx] >= a:
                count += 1
            idx += 1
        thr_samples.append(count / window)

    return util_samples, thr_samples


def export_bm_series_to_wide_csv(system, B, out_csv,
                                 burn_in=0, k_max=None):
    """
    Crea un CSV 'wide' con colonne per CENTRO:
      - RT batch means: web_rt, spike{i}_rt, mit_rt   (N O  system_rt)
      - Util/Thr per batch: web_util, web_thr, spike{i}_util, spike{i}_thr, mit_util, mit_thr
    Tutte le colonne hanno lunghezza comune = min #batch disponibile tra le serie incluse.
    Ritorna: (path_csv, lista_colonne, k_batch_comuni)
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    center = system.mitigation_manager.center

    # Serie per-job dei tempi di risposta (per centro)
    web_rt    = list(system.web_server.completed_jobs)
    mit_rt    = list(center.completed_jobs)
    spike_rts = [list(s.completed_jobs) for s in system.spike_servers]

    # RT per centro 
    items_rt = [("web_rt", web_rt), ("mit_rt", mit_rt)]
    for i, r in enumerate(spike_rts):
        items_rt.insert(1, (f"spike{i}_rt", r))  

    # Calcola RT batch-means per ogni centro
    bm_dict = {}
    lengths = []
    for name, series in items_rt:
        bm = make_batch_means_series(series, b=B, burn_in=burn_in, k_max=k_max)
        bm_dict[name] = bm
        lengths.append(len(bm))

    # Util/Thr per ogni centro 
    ordered_cols = []
    now = system.env.now
    # Web
    web_util, web_thr = util_thr_per_batch(
        system.web_server.busy_periods, system.web_server.completion_times,
        B=B, burn_in=burn_in, k_max=k_max, tmax=now
    )
    bm_dict["web_util"] = web_util; bm_dict["web_thr"] = web_thr
    lengths += [len(web_util), len(web_thr)]

    # Spike-i
    for i, s in enumerate(system.spike_servers):
        su, st = util_thr_per_batch(
            s.busy_periods, s.completion_times,
            B=B, burn_in=burn_in, k_max=k_max, tmax=now
        )
        bm_dict[f"spike{i}_util"] = su; bm_dict[f"spike{i}_thr"] = st
        lengths += [len(su), len(st)]

    # Mitigation
    mit_util, mit_thr = util_thr_per_batch(
        center.busy_periods, center.completion_times,
        B=B, burn_in=burn_in, k_max=k_max, tmax=now
    )
    bm_dict["mit_util"] = mit_util; bm_dict["mit_thr"] = mit_thr
    lengths += [len(mit_util), len(mit_thr)]

    # Lunghezza comune
    if not bm_dict:
        raise ValueError("Nessuna metrica inclusa: controlla include_metrics o i dati disponibili.")
    k_common = min(len(v) for v in bm_dict.values())

    # Ordine colonne: Web, Spike-*, Mitigation
    if "web_rt" in bm_dict:    ordered_cols.append("web_rt")
    if "web_util" in bm_dict:  ordered_cols += ["web_util", "web_thr"]
    i = 0
    while True:
        had_any = False
        rt_k  = f"spike{i}_rt"
        ut_k  = f"spike{i}_util"
        th_k  = f"spike{i}_thr"
        if rt_k in bm_dict:
            ordered_cols.append(rt_k); had_any = True
        if ut_k in bm_dict and th_k in bm_dict:
            ordered_cols += [ut_k, th_k]; had_any = True
        if not had_any:
            break
        i += 1
    if "mit_rt" in bm_dict:    ordered_cols.append("mit_rt")
    if "mit_util" in bm_dict:  ordered_cols += ["mit_util", "mit_thr"]

    # Costruisci e salva CSV
    df = pd.DataFrame({c: np.asarray(bm_dict[c], dtype=float)[:k_common] for c in ordered_cols})
    df.to_csv(out_csv, index=False)
    return out_csv, ordered_cols, k_common

def calculate_autocorrelation(data, K_LAG=50):
    """
    Calcola l'autocorrelazione fino a K_LAG (come indicato dalla libreria acs.py).
    Ritorna: mean, stdev, autocorr(list di length K_LAG), n
    """
    SIZE = K_LAG + 1
    x = np.asarray(data, dtype=float)
    n = int(x.size)
    if n <= K_LAG:
        raise ValueError(f"length of data must be greater than K (n={n}, K={K_LAG})")

    hold = [0.0] * SIZE
    cosum = [0.0] * SIZE
    sum_x = 0.0
    p = 0

    hold[:SIZE] = x[:SIZE].tolist()
    sum_x += float(np.sum(x[:SIZE]))


    for i in range(SIZE, n):
        xi = float(x[i])
        for j in range(SIZE):
            cosum[j] += hold[p] * hold[(p + j) % SIZE]
        hold[p] = xi
        sum_x += xi
        p = (p + 1) % SIZE

    for _ in range(n, n + SIZE):
        for j in range(SIZE):
            cosum[j] += hold[p] * hold[(p + j) % SIZE]
        hold[p] = 0.0
        p = (p + 1) % SIZE

    mean = sum_x / n
    for j in range(SIZE):
        cosum[j] = (cosum[j] / (n - j)) - (mean * mean)

    var0 = cosum[0]
    if var0 <= 0:
        stdev = 0.0
        autocorr = [0.0] * K_LAG
    else:
        stdev = math.sqrt(var0)
        autocorr = [cosum[j] / var0 for j in range(1, SIZE)]

    return mean, stdev, autocorr, n

def _batch_intervals_from_completions(completion_times, B, burn_in=0, k_max=None):
    ct = np.asarray(completion_times, dtype=float)
    if burn_in > 0 and ct.size > burn_in:
        ct = ct[burn_in:]
    n = ct.size
    if n < B:
        raise ValueError(f"Not enough completions after burn-in: n={n} < B={B}")
    k = n // B
    if k_max is not None:
        k = min(k, int(k_max))
    starts = ct[0 : k * B : B]
    ends   = ct[B - 1 : k * B : B]
    return np.column_stack([starts, ends])  # (k,2)

def _closed_periods(periods, tmax):
    out = []
    for s, e in periods:
        if e is None:
            e = float(tmax)
        out.append((float(s), float(e)))
    return out

def util_thr_per_batch(busy_periods, completion_times, B, burn_in=0, k_max=None, tmax=None):
    """
    Per un dato CENTRO:
      - throughput_batch = B / Δt, con Δt = [1°..ultimo] completion del batch di quel centro
      - utilization_batch = busy_overlap(Δt) / Δt, usando busy_periods del centro
    Ritorna (util_series, thr_series) allineati per batch.
    """
    intervals = _batch_intervals_from_completions(completion_times, B, burn_in=burn_in, k_max=k_max)
    if tmax is None:
        tmax = intervals[-1, 1]
    periods = _closed_periods(busy_periods, tmax)

    util, thr = [], []
    eps = np.finfo(float).eps
    for a, b_end in intervals:
        dt = max(b_end - a, eps)
        busy = 0.0
        for s, e in periods:
            if e <= a or s >= b_end:
                continue
            busy += max(0.0, min(e, b_end) - max(s, a))
        util.append(busy / dt)
        thr.append(B / dt)   
    return np.asarray(util), np.asarray(thr)


def print_autocorrelation(file_path,
                          columns=None,
                          K_LAG=50,
                          threshold=0.2,
                          save_csv=None):
    """
    Legge un CSV wide, calcola ACF fino a K_LAG per ciascuna colonna.
    - columns: lista di colonne da analizzare (default: tutte numeriche)
    - threshold: stampa PASS/FAIL su |rho_1| < threshold
    - save_csv: se non None, salva un CSV con mean/stdev/rho_1..rho_K
    Ritorna: DataFrame con risultati
    """
    df = pd.read_csv(file_path)
    if columns is None:
        # solo colonne numeriche
        columns = [c for c in df.columns if np.issubdtype(df[c].dropna().dtype, np.number)]

    results = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        try:
            mean, stdev, ac, n = calculate_autocorrelation(s, K_LAG=K_LAG)
            rho1 = ac[0] if ac else float("nan")
            ok = (abs(rho1) < threshold)
            print(f"{col}: rho_1={rho1:.3f}  → {'PASS' if ok else 'FAIL'}  (n={n})")
            row = {
                "metric": col, "n": n, "mean": mean, "stdev": stdev,
                "rho_1": rho1, "pass_thr": int(ok)
            }
            
            for k in range(1, K_LAG + 1):
                rk = ac[k-1] if k-1 < len(ac) else np.nan
                row[f"rho_{k}"] = rk
            results.append(row)
        except Exception as e:
            print(f"Error {col}: {e}")

    out = pd.DataFrame(results)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        out.to_csv(save_csv, index=False)
    return out

