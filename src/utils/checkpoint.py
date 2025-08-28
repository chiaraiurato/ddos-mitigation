import numpy as np

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
    c = 0
    for x in times:
        if x <= t: c += 1
        else: break
    return c

def _throughput_upto(completion_times, t):
    if t <= 0:
        return 0.0
    return _count_upto(completion_times, t) / t

def _rt_mean_upto(rt_samples, completion_times, t):
    n = _count_upto(completion_times, t)
    if n == 0:
        return None
    return float(np.mean(rt_samples[:n]))
