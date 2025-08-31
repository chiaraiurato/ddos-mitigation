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
    means, _ = _compute_batch_means(data, b, burn_in=burn_in, k_max=k_max, estimate_ci=False)
    return means

def batch_means(data, batch_size=None, n_batches=None, confidence=None, burn_in=0):
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


def export_bm_series_to_wide_csv(system, B, out_csv, burn_in=0, k_max=None, include_analysis_center=False):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    center = system.mitigation_manager.center

    series_map = {
        "web_rt": list(system.web_server.completed_jobs),
        "mit_rt": list(center.completed_jobs),
    }
    for i, s in enumerate(system.spike_servers):
        series_map[f"spike{i}_rt"] = list(s.completed_jobs)

    def n_eff(seq):
        return max(0, len(seq) - burn_in)

    skipped = []
    bm_dict = {}

    for name, seq in series_map.items():
        if n_eff(seq) >= B:
            bm_dict[name] = make_batch_means_series(seq, b=B, burn_in=burn_in, k_max=k_max)
        else:
            skipped.append((name, n_eff(seq)))

    if skipped:
        print("[export] Skipping RT metrics with insufficient completions for B:")
        for n, m in skipped:
            print(f"  - {n}: n_eff={m} < B={B}")

    now = system.env.now

    def add_util_thr(prefix, busy_periods, comp_times):
        if (len(comp_times) - burn_in) >= B:
            u, t = util_thr_per_batch(busy_periods, comp_times, B=B, burn_in=burn_in, k_max=k_max, tmax=now)
            bm_dict[f"{prefix}_util"] = u
            bm_dict[f"{prefix}_thr"]  = t
        else:
            print(f"[export] Skipping {prefix}_util/thr (n_eff < B)")

    add_util_thr("web", system.web_server.busy_periods, system.web_server.completion_times)
    for i, s in enumerate(system.spike_servers):
        add_util_thr(f"spike{i}", s.busy_periods, s.completion_times)
    add_util_thr("mit", center.busy_periods, center.completion_times)

    if include_analysis_center:
        ac = system.mitigation_manager.analysis_center
        if ac is None:
            print("[export] Analysis center not present → skipped.")
        else:

            if hasattr(ac, "update"):
                try:
                    ac.update(now)
                except Exception:
                    pass

            cj = list(getattr(ac, "completed_jobs", []))
            ct = list(getattr(ac, "completion_times", []))
            bp = list(getattr(ac, "busy_periods", []))

            if n_eff(cj) >= B:
                bm_dict["ana_rt"] = make_batch_means_series(cj, b=B, burn_in=burn_in, k_max=k_max)
            else:
                print("[export] Skipping ana_rt (n_eff < B)")

            if n_eff(ct) >= B:
                u, t = util_thr_per_batch(bp, ct, B=B, burn_in=burn_in, k_max=k_max, tmax=now)
                bm_dict["ana_util"], bm_dict["ana_thr"] = u, t
            else:
                print("[export] Skipping ana_util/thr (n_eff < B)")

    if not bm_dict:
        raise ValueError("Nessuna colonna esportabile: tutti i centri sono sotto B.")

    
    lengths = {c: len(bm_dict[c]) for c in bm_dict.keys()}
    k_rows = max(lengths.values())

    ordered = []
    if "web_rt" in bm_dict:   ordered.append("web_rt")
    if "web_util" in bm_dict: ordered += ["web_util", "web_thr"]

    i = 0
    while True:
        had = False
        if f"spike{i}_rt" in bm_dict:
            ordered.append(f"spike{i}_rt"); had = True
        if f"spike{i}_util" in bm_dict and f"spike{i}_thr" in bm_dict:
            ordered += [f"spike{i}_util", f"spike{i}_thr"]; had = True
        if not had:
            break
        i += 1

    if "mit_rt" in bm_dict:   ordered.append("mit_rt")
    if "mit_util" in bm_dict: ordered += ["mit_util", "mit_thr"]

    if "ana_rt" in bm_dict:   ordered.append("ana_rt")
    if "ana_util" in bm_dict: ordered += ["ana_util", "ana_thr"]

    def _pad_to(arr, L):
        a = np.asarray(arr, dtype=float)
        if a.size < L:
            pad = np.full(L - a.size, np.nan, dtype=float)
            a = np.concatenate([a, pad])
        return a[:L]

    data = {c: _pad_to(bm_dict[c], k_rows) for c in ordered}

    df = pd.DataFrame(data)
    df.to_csv(out_csv, index=False)
    return out_csv, ordered, k_rows




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


def acf(data):
    x = np.asarray(data, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n <= 1:
        return np.nan, n
    mu = np.mean(x)
    num = np.sum((x[:-1] - mu) * (x[1:] - mu))
    den = np.sum((x - mu) ** 2)
    if den == 0:
        return 0.0, n
    return num / den, n

def print_autocorrelation(file_path, columns=None, save_csv=None):
    
    df = pd.read_csv(file_path)

    if columns is None:
        columns = [c for c in df.columns
                   if np.issubdtype(pd.to_numeric(df[c], errors="coerce").dtype, np.number)]

    results = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce").to_numpy()
        s = s[~np.isnan(s)]
        try:
            rho1, n = acf(s)
            if n <= 1 or np.isnan(rho1):
                raise ValueError(f"n={n} <= 1")
            K = 128
            thr = 2.0 / np.sqrt(K)          # (Chatfield)
            ok  = (abs(rho1) < thr)

            print(f"{col}: rho_1={rho1:.3f}  (|rho1|<{thr:.3f}) → {'PASS' if ok else 'FAIL'}")
            results.append({"metric": col, "n": n, "rho_1": rho1, "thr": thr, "pass_thr": int(ok)})

        except Exception as e:
            print(f"SKIP {col}: {e}")

    out = pd.DataFrame(results)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        out.to_csv(save_csv, index=False)
    return out


