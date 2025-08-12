import numpy as np
from scipy.stats import t

def batch_means(data, batch_size):
    n = len(data)
    if n < batch_size:
        raise ValueError("Not enough data for batch means")

    k = n // batch_size  # numero di batch interi
    batches = np.array_split(data[:k * batch_size], k)
    means = np.array([np.mean(b) for b in batches])

    mean = np.mean(means)
    std_err = np.std(means, ddof=1) / np.sqrt(k)
    ci_half_width = t.ppf(0.975, df=k - 1) * std_err

    return mean, ci_half_width

def window_util_thr(busy_periods, completion_times, window, now):
    """
    Calcola (utilization, throughput) su finestre non sovrapposte di ampiezza `window`
    nell'intervallo [0, now), usando:
      - busy_periods: lista di (start, end)
      - completion_times: lista di tempi assoluti di completamento (monotona crescente non obbligatoria)
    Ritorna: (util_samples: list[float], thr_samples: list[float])
    """
    if window <= 0 or now <= 0:
        return [], []

    nwin = int(now // window)
    if nwin <= 0:
        return [], []

    util_samples = []
    thr_samples = []

    comp = completion_times
    ncomp = len(comp)
    idx = 0

    for k in range(nwin):
        a = k * window
        b = a + window

        # Utilization: frazione di tempo busy nella finestra
        busy_in_window = 0.0
        for (s, e) in busy_periods:
            if e <= a or s >= b:
                continue
            start = max(s, a)
            end = min(e, b)
            if end > start:
                busy_in_window += end - start
        util_samples.append(busy_in_window / window)

        # Throughput: completamenti in [a,b) / window
        count = 0
        while idx < ncomp and comp[idx] < b:
            if comp[idx] >= a:
                count += 1
            idx += 1
        thr_samples.append(count / window)

    return util_samples, thr_samples
