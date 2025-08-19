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

def _safe_bm_series(series, b, burn_in=0, k_max=None):
    """Restituisce la serie batch-means oppure None se non si riescono a formare ≥2 batch."""
    try:
        bm = make_batch_means_series(series, b=b, burn_in=burn_in, k_max=k_max)
        return bm if len(bm) >= 2 else None
    except Exception:
        return None

def export_bm_series_to_wide_csv(system, B, out_csv,
                                 burn_in=0, k_max=None,
                                 k_min_common=None,  # <-- NOVITÀ: minimo richiesto per l'export
                                 verbose=True):
    """
    Crea un CSV 'wide' con colonne per CENTRO:
      - RT batch means: web_rt, spike{i}_rt, mit_rt
      - Util/Thr per batch: web_util, web_thr, spike{i}_util, spike{i}_thr, mit_util, mit_thr
    Includo SOLO le serie con almeno 2 batch e, se k_min_common è impostato,
    SOLO quelle con almeno k_min_common punti (prima di allineare alla k_common).
    Ritorna: (path_csv, lista_colonne, k_batch_comuni)
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    center = system.mitigation_manager.center
    now = system.env.now

    bm_dict = {}  # colonna -> ndarray di length K>=2

    # =======================
    # 1) RT per centro (BM)
    # =======================
    web_rt = list(system.web_server.completed_jobs)
    mit_rt = list(center.completed_jobs)
    spike_rts = [list(s.completed_jobs) for s in system.spike_servers]

    web_rt_bm = _safe_bm_series(web_rt, B, burn_in, k_max)
    if web_rt_bm is not None:
        bm_dict["web_rt"] = web_rt_bm

    for i, r in enumerate(spike_rts):
        s_bm = _safe_bm_series(r, B, burn_in, k_max)
        if s_bm is not None:
            bm_dict[f"spike{i}_rt"] = s_bm

    mit_rt_bm = _safe_bm_series(mit_rt, B, burn_in, k_max)
    if mit_rt_bm is not None:
        bm_dict["mit_rt"] = mit_rt_bm

    # ============================================
    # 2) Utilization / Throughput per-batch (B fisso)
    # ============================================
    def _safe_util_thr(busy_periods, completion_times):
        try:
            u, t = util_thr_per_batch(busy_periods, completion_times,
                                      B=B, burn_in=burn_in, k_max=k_max, tmax=now)
            u = np.asarray(u); t = np.asarray(t)
            if len(u) < 2 or len(t) < 2:
                return None, None
            return u, t
        except Exception:
            return None, None

    u, t = _safe_util_thr(system.web_server.busy_periods, system.web_server.completion_times)
    if u is not None:
        bm_dict["web_util"] = u; bm_dict["web_thr"] = t

    for i, s in enumerate(system.spike_servers):
        u, t = _safe_util_thr(s.busy_periods, s.completion_times)
        if u is not None:
            bm_dict[f"spike{i}_util"] = u; bm_dict[f"spike{i}_thr"] = t

    u, t = _safe_util_thr(center.busy_periods, center.completion_times)
    if u is not None:
        bm_dict["mit_util"] = u; bm_dict["mit_thr"]  = t

    # ============================================
    # 3) Ordering colonne
    # ============================================
    ordered_cols = []
    if "web_rt"   in bm_dict: ordered_cols.append("web_rt")
    if "web_util" in bm_dict: ordered_cols += ["web_util", "web_thr"]

    spike_ids = sorted({
        int(k.split('_')[0][5:]) for k in bm_dict.keys()
        if k.startswith("spike")
    })
    for i in spike_ids:
        if f"spike{i}_rt"   in bm_dict: ordered_cols.append(f"spike{i}_rt")
        if f"spike{i}_util" in bm_dict and f"spike{i}_thr" in bm_dict:
            ordered_cols += [f"spike{i}_util", f"spike{i}_thr"]

    if "mit_rt"   in bm_dict: ordered_cols.append("mit_rt")
    if "mit_util" in bm_dict: ordered_cols += ["mit_util", "mit_thr"]

    if not ordered_cols:
        raise ValueError("Nessuna metrica con almeno 2 batch (B fisso). Riduci B o prolunga la simulazione.")

    # ============================================
    # 4) Filtro per k_min_common (es. K_LAG+1)
    #    → rimuove le colonne troppo corte
    # ============================================
    if k_min_common is not None:
        short = [c for c in ordered_cols if len(bm_dict[c]) < int(k_min_common)]
        if verbose:
            for c in short:
                print(f"[WARN] drop '{c}': only {len(bm_dict[c])} batch points < required {k_min_common}")
        ordered_cols = [c for c in ordered_cols if len(bm_dict[c]) >= int(k_min_common)]
        for c in short:
            bm_dict.pop(c, None)

    if not ordered_cols:
        raise ValueError(f"Nessuna colonna con almeno {k_min_common} batch: aumenta durata o riduci B/k_min_common.")

    # ============================================
    # 5) Allineamento sulla min rimanente e salvataggio
    # ============================================
    k_common = min(len(bm_dict[c]) for c in ordered_cols)
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
      - utilization_batch = busy_overlap(Δt) / Δt
    Richiede B fisso. Se non ci sono almeno 2 batch (n < 2*B), ritorna serie vuote.
    """
    ct = np.asarray(completion_times, dtype=float)
    if burn_in > 0 and ct.size > burn_in:
        ct = ct[burn_in:]
    n = int(ct.size)

    # Con B fisso vogliamo almeno 2 batch ⇒ n ≥ 2*B
    if n < 2 * B:
        return np.asarray([]), np.asarray([])

    # Da qui in poi è sicuro creare gli intervalli
    intervals = _batch_intervals_from_completions(
        completion_times, B, burn_in=burn_in, k_max=k_max
    )

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
        thr.append(B / dt)  # B rimane quello fissato dall’utente

    return np.asarray(util), np.asarray(thr)



def print_autocorrelation(file_path,
                          columns=None,
                          K_LAG=50,
                          threshold=0.2,
                          save_csv=None):
    """
    Legge un CSV wide, calcola ACF fino a K_LAG per ciascuna colonna.
    - columns: lista di colonne da analizzare (default: tutte numeriche)
    - threshold: stampa PASS/FAIL su rho_1 < threshold
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
            ok = (rho1) < threshold
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

