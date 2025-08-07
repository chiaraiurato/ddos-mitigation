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

def compute_throughput(timestamps, window=1.0):
    if not timestamps:
        return []

    start = timestamps[0]
    end = timestamps[-1]
    windows = np.arange(start, end, window)
    
    counts = []
    idx = 0
    for w_start in windows:
        w_end = w_start + window
        count = 0
        while idx < len(timestamps) and timestamps[idx] < w_end:
            count += 1
            idx += 1
        counts.append(count / window)
    return counts
