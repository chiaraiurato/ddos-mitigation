import numpy as np
# === CHECKPOINT HELPERS ===
def _utilization_upto(busy_periods, t):
    """Utilization cumulativa fino al tempo t."""
    if t <= 0:
        return 0.0
    busy = 0.0
    for (start, end) in busy_periods:
        if start >= t:
            break
        e = t if (end is None or end > t) else end
        if e > start:
            busy += (e - start)
    return busy / t

def _count_upto(times, t):
    """Conta quanti istanti in 'times' sono <= t (lista ordinata)."""
    # Se times è ordinata (lo è per i completamenti), una scansione è sufficiente.
    c = 0
    for x in times:
        if x <= t: c += 1
        else: break
    return c

def _throughput_upto(completion_times, t):
    """Throughput cumulativo (completion rate) fino a t."""
    if t <= 0:
        return 0.0
    return _count_upto(completion_times, t) / t

def _rt_mean_upto(rt_samples, completion_times, t):
    """
    Media dei response time dei job completati entro t.
    Si assume che rt_samples[i] corrisponda a completion_times[i].
    """
    n = _count_upto(completion_times, t)
    if n == 0:
        return None
    return float(np.mean(rt_samples[:n]))
