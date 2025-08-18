import numpy as np
from library.rvms import idfStudent
from engineering.costants import BATCH_SIZE, N_BATCH, CONFIDENCE_LEVEL

def make_batch_means_series(data, b, burn_in=0, k_max=None):
    """
    Restituisce la SERIE delle medie di batch.
    - data: array-like dei tempi di risposta per-job
    - b: batch size s
    - burn_in: scarta i primi 'burn_in' completamenti prima di batchizzare
    - k_max: opzionale, limita il numero di batch (per file più piccoli)
    """
    if data is None:
        raise ValueError("data=None")
    x = np.asarray(list(data), dtype=float)
    if burn_in > 0 and x.size > burn_in:
        x = x[burn_in:]
    n = x.size
    if n < b:
        raise ValueError(f"Not enough samples after burn-in: n={n} < b={b}")
    k = n // b
    if k_max is not None:
        k = min(k, int(k_max))
    trimmed = x[:k * b]
    means = trimmed.reshape(k, b).mean(axis=1)
    return means 

def _make_batches(data, b, k_max=None):
    n = len(data)
    if n < b:
        raise ValueError(f"Not enough data for batch means: n={n} < b={b}")
    k = n // b
    if k_max is not None:
        k = min(k, k_max)
    if k < 2:
        raise ValueError(f"Need at least 2 batches, got k={k} (n={n}, b={b}).")

    trimmed = np.asarray(data[:k * b], dtype=float)
    batches = np.array_split(trimmed, k)     
    means = np.array([np.mean(bi) for bi in batches], dtype=float)
    return means, k

def batch_means(data, batch_size=None, n_batches=None, confidence=None):
    """
    Algoritmo (Batch Means):
      - b = batch_size (default: BATCH_SIZE)
      - k <= n_batches (default: N_BATCH)
      - livello di confidenza (default: CONFIDENCE_LEVEL)
    Ritorna: (xbar, half_width)
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    if n_batches is None:
        n_batches = N_BATCH
    if confidence is None:
        confidence = CONFIDENCE_LEVEL

    means, k = _make_batches(data, batch_size, k_max=n_batches)

    xbar = float(np.mean(means))
    s = float(np.std(means, ddof=1))
    alpha = 1.0 - float(confidence)
    u = 1.0 - alpha / 2.0
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


def batch_means_cut_burn_in(data, batch_size=None, n_batches=None, confidence=None, burn_in=0):
    """
    Scarta i primi 'burn_in' completamenti.
    """
    if data is None:
        raise ValueError("data=None")
    if burn_in > 0 and len(data) > burn_in:
        data = data[burn_in:]
    return batch_means(
        data,
        batch_size=batch_size,
        n_batches=n_batches,
        confidence=confidence
    )
