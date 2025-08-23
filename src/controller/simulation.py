import os
import simpy
import numpy as np
import pandas as pd

from library.rngs import random, plantSeeds, getSeed, selectStream
from engineering.costants import *
from engineering.distributions import get_interarrival_time
from utils.csv_writer import append_row_stable, validation_fieldnames, transitory_fieldnames, infinite_fieldnames
from model.job import Job
from model.processor_sharing_server import ProcessorSharingServer
from model.mitigation_manager import MitigationManager
from utils.checkpoint import _rt_mean_upto, _utilization_upto, _throughput_upto
from engineering.statistics import (
    batch_means, export_bm_series_to_wide_csv, print_autocorrelation,
    util_thr_per_batch, window_util_thr
)

# ---------------------------------------------------------------------
# Helpers per CSV "wide" grafici: colonne richieste e densificazione (≥64 righe)
# ---------------------------------------------------------------------
REQUIRED_GRAPH_COLS = [
    "web_rt", "web_util", "web_thr",
    "spike0_rt", "spike0_util", "spike0_thr",
    "spike1_rt", "spike1_util", "spike1_thr",
    "spike2_rt", "spike2_util", "spike2_thr",
    "spike3_rt", "spike3_util", "spike3_thr",
    "mit_rt", "mit_util", "mit_thr",
    "ana_rt", "ana_util", "ana_thr",
]

def _coerce_and_order_graph_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in REQUIRED_GRAPH_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in REQUIRED_GRAPH_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[REQUIRED_GRAPH_COLS]

def _densify_to_min_rows(df: pd.DataFrame, min_rows: int = 64) -> pd.DataFrame:
    """
    Garantisce almeno 'min_rows' righe numeriche.
    - Se < min_rows: upsampling lineare a min_rows
    - Se >= min_rows: lascia invariato
    """
    df = df.copy()
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.ffill().bfill()
    df = df.fillna(0.0)

    n = len(df)
    if n == 0:
        raise ValueError("Wide CSV vuoto: impossibile densificare.")
    if n >= min_rows:
        return df.reset_index(drop=True)

    # Upsampling lineare a min_rows
    src = df.reset_index(drop=True)
    xi = np.linspace(0.0, float(n - 1), min_rows)
    out = pd.DataFrame(index=range(min_rows))
    for col in src.columns:
        y = src[col].to_numpy(dtype=float)
        out[col] = np.interp(xi, np.arange(n, dtype=float), y)
    return out

def _fix_wide_csv_on_disk(csv_path: str, min_rows: int = 64):
    """Rende il CSV wide adatto ai grafici: colonne richieste + ≥64 righe."""
    df = pd.read_csv(csv_path)
    df = _coerce_and_order_graph_df(df)
    df = _densify_to_min_rows(df, min_rows=min_rows)
    df.to_csv(csv_path, index=False)

# ---------------------------------------------------------------------
# fallback per costanti scenario transitorio / finito (se rinominate altrove)
# ---------------------------------------------------------------------
try:
    STOP_TRANSITORY = STOP_CONDITION_TRANSITORY
except NameError:
    try:
        STOP_TRANSITORY = STOP_CONDITION_TRANSITORY
    except NameError:
        STOP_TRANSITORY = STOP_CONDITION_FINITE_SIMULATION

try:
    CHECKPOINT_TRANSITORY = CHECKPOINT_TIME_TRANSITORY
except NameError:
    try:
        CHECKPOINT_TRANSITORY = CHECKPOINT_TIME_TRANSITORY
    except NameError:
        CHECKPOINT_TRANSITORY = CHECKPOINT_TIME_FINITE_SIMULATION


class DDoSSystem:
    def __init__(self, env, mode, arrival_p, arrival_l1, arrival_l2, variant):
        self.env = env
        self.mode = mode
        self.variant = variant

        self.arrival_p = arrival_p
        self.arrival_l1 = arrival_l1
        self.arrival_l2 = arrival_l2

        self.web_server = ProcessorSharingServer(env, "Web")
        self.spike_servers = [ProcessorSharingServer(env, "Spike-0")]

        # Metriche globali
        self.metrics = {
            "mitigation_completions": 0,
            "total_arrivals": 0,
            "false_positives": 0,
            "legal_arrivals": 0,
            "illegal_arrivals": 0,
            "false_positives_legal": 0,
        }

        self.mitigation_manager = MitigationManager(
            env, self.web_server, self.spike_servers, self.metrics, self.mode, variant=self.variant
        )

        self.env.process(self.arrival_process(mode))

    # -----------------------------------------------------------------
    # Helper: busy periods del centro di Analisi (UNION, solo fallback)
    # -----------------------------------------------------------------
    def _get_analysis_busy_periods(self, ac, now):
        periods = getattr(ac, "busy_periods", None)
        if periods is not None:
            out = []
            for (a, b) in periods:
                if b is None:
                    b = now
                if b > a:
                    out.append((a, b))
            return out

        core_lists = getattr(ac, "core_busy_periods", [])
        flat = []
        for lst in core_lists:
            for (a, b) in lst:
                if b is None:
                    b = now
                if b > a:
                    flat.append((a, b))
        if not flat:
            return []

        flat.sort(key=lambda x: x[0])
        merged = [list(flat[0])]
        for a, b in flat[1:]:
            if a > merged[-1][1]:
                merged.append([a, b])
            else:
                if b > merged[-1][1]:
                    merged[-1][1] = b
        return [(a, b) for a, b in merged]

    # -----------------------------------------------------------------
    # Helper: utilizzo corretto (single-server)
    # -----------------------------------------------------------------
    def _single_util(self, server, now):
        return getattr(server, "busy_time", 0.0) / max(now, 1e-12)

    # -----------------------------------------------------------------
    # Helper: utilizzo Analysis rate-based (coerente con Markov)
    # -----------------------------------------------------------------
    def _analysis_util_rate_based(self, ac, now):
        c = getattr(ac, "num_cores", getattr(ac, "cores", 1))
        if c is None or c <= 0:
            c = 1
        comps = int(getattr(ac, "total_completions", 0))
        thr = comps / max(now, 1e-12)
        if getattr(ac, "completed_jobs", None) and ac.completed_jobs:
            avg_rt = float(np.mean(ac.completed_jobs))
        else:
            avg_rt = 0.0
        return (thr * avg_rt) / c

    def _analysis_util_rate_based_upto(self, ac, when):
        c = getattr(ac, "num_cores", getattr(ac, "cores", 1))
        if c is None or c <= 0:
            c = 1
        thr_upto = _throughput_upto(ac.completion_times, when)
        avg_rt_upto = _rt_mean_upto(ac.completed_jobs, ac.completion_times, when) or 0.0
        return (thr_upto * avg_rt_upto) / c

    def _window_util_thr_analysis_rate_based(self, ac, window, tmax):
        c = getattr(ac, "num_cores", getattr(ac, "cores", 1))
        if c is None or c <= 0:
            c = 1

        times = list(getattr(ac, "completion_times", []))
        rts = list(getattr(ac, "completed_jobs", []))

        # allineamento: rts[k] completa in times[k]
        n = min(len(times), len(rts))
        pairs = [(times[k], rts[k]) for k in range(n) if times[k] <= tmax]

        util_samples, thr_samples = [], []
        t = window
        while t <= tmax + 1e-12:
            w_start, w_end = t - window, t
            in_win = [rt for (tt, rt) in pairs if (w_start < tt <= w_end)]
            comps = len(in_win)
            thr = comps / window
            avg_rt_w = (sum(in_win) / comps) if comps > 0 else 0.0
            util_w = (thr * avg_rt_w) / c
            util_samples.append(util_w)
            thr_samples.append(thr)
            t += window
        return util_samples, thr_samples

    # ---------------------- SNAPSHOT (orizzonte finito) ----------------------
    def snapshot(self, when, replica_id=None):
        self.web_server.update(when)
        for s in self.spike_servers:
            s.update(when)
        center = self.mitigation_manager.center
        center.update(when)

        if self.mitigation_manager.analysis_center is not None:
            self.mitigation_manager.analysis_center.update(when)

        row = {
            "replica": int(replica_id) if replica_id is not None else None,
            "time": float(when),

            # Web
            "web_rt_mean": _rt_mean_upto(self.web_server.completed_jobs,
                                         self.web_server.completion_times, when),
            "web_util": _utilization_upto(self.web_server.busy_periods, when),
            "web_throughput": _throughput_upto(self.web_server.completion_times, when),

            # Mitigation
            "mit_rt_mean": _rt_mean_upto(center.completed_jobs, center.completion_times, when),
            "mit_util": _utilization_upto(center.busy_periods, when),
            "mit_throughput": _throughput_upto(center.completion_times, when),
        }

        if self.mitigation_manager.analysis_center is not None:
            ac = self.mitigation_manager.analysis_center
            row["analysis_rt_mean"] = _rt_mean_upto(ac.completed_jobs, ac.completion_times, when)
            row["analysis_util"] = self._analysis_util_rate_based_upto(ac, when)
            row["analysis_throughput"] = _throughput_upto(ac.completion_times, when)

        row.update({
            "arrivals_so_far": int(self.metrics["total_arrivals"]),
            "false_positives_so_far": int(self.metrics.get("false_positives", 0)),
            "mitigation_completions_so_far": int(self.metrics.get("mitigation_completions", 0)),
            "spikes_count": int(len(self.spike_servers)),
        })

        for i in range(len(self.spike_servers)):
            s = self.spike_servers[i]
            row[f"spike{i}_rt_mean"] = _rt_mean_upto(s.completed_jobs, s.completion_times, when)
            row[f"spike{i}_util"] = _utilization_upto(s.busy_periods, when)
            row[f"spike{i}_throughput"] = _throughput_upto(s.completion_times, when)

        for i in range(len(self.spike_servers), MAX_SPIKE_NUMBER):
            row[f"spike{i}_rt_mean"] = None
            row[f"spike{i}_util"] = None
            row[f"spike{i}_throughput"] = None

        tot_arr = max(1, self.metrics["total_arrivals"])

        proc_leg = int(self.metrics.get("processed_legal", 0))
        proc_illeg = int(self.metrics.get("processed_illegal", 0))

        web_leg = int(self.web_server.legal_completions)
        web_illeg = int(self.web_server.illegal_completions)
        spike_leg = sum(int(s.legal_completions) for s in self.spike_servers)
        spike_illeg = sum(int(s.illegal_completions) for s in self.spike_servers)

        comp_leg = web_leg + spike_leg
        comp_illeg = web_illeg + spike_illeg
        comp_tot = max(1, comp_leg + comp_illeg)

        row["illegal_share"] = self.metrics["illegal_arrivals"] / tot_arr
        row["processed_legal_share"] = proc_leg / tot_arr
        row["processed_illegal_share"] = proc_illeg / tot_arr
        row["completed_legal_share"] = comp_leg / tot_arr
        row["completed_illegal_share"] = comp_illeg / tot_arr
        row["completed_legal_of_completed_share"] = comp_leg / comp_tot
        row["completed_illegal_of_completed_share"] = comp_illeg / comp_tot

        sys_rts = list(self.web_server.completed_jobs)
        sys_times = list(self.web_server.completion_times)
        for s in self.spike_servers:
            sys_rts.extend(s.completed_jobs)
            sys_times.extend(s.completion_times)
        row["system_rt_mean"] = _rt_mean_upto(sys_rts, sys_times, when)

        return row

    # ---------------------- ARRIVALS ----------------------
    def arrival_process(self, mode):
        if mode == "verification":
            arrivals = N_ARRIVALS_VERIFICATION
            interarrival_mode = "verification"
            stop_on_arrivals = True
        elif mode in ("transitory", "finite simulation"):
            interarrival_mode = "standard"
            stop_on_arrivals = False
        elif mode == "infinite simulation":
            arrivals = N_ARRIVALS_BATCH_MEANS
            interarrival_mode = "standard"
            stop_on_arrivals = True
        else:  # "standard"
            arrivals = N_ARRIVALS
            interarrival_mode = "standard"
            stop_on_arrivals = True

        if not stop_on_arrivals:
            while True:
                interarrival_time = get_interarrival_time(
                    interarrival_mode, self.arrival_p, self.arrival_l1, self.arrival_l2
                )
                yield self.env.timeout(interarrival_time)
                self.metrics["total_arrivals"] += 1
                arrival_time = self.env.now
                is_legal = (random() < P_LECITO)
                if is_legal: self.metrics["legal_arrivals"] += 1
                else:        self.metrics["illegal_arrivals"] += 1
                job = Job(self.metrics["total_arrivals"], arrival_time, None, is_legal)
                self.mitigation_manager.handle_job(job)
        else:
            while self.metrics["total_arrivals"] < arrivals:
                interarrival_time = get_interarrival_time(
                    interarrival_mode, self.arrival_p, self.arrival_l1, self.arrival_l2
                )
                yield self.env.timeout(interarrival_time)
                self.metrics["total_arrivals"] += 1
                arrival_time = self.env.now
                is_legal = (random() < P_LECITO)
                if is_legal: self.metrics["legal_arrivals"] += 1
                else:        self.metrics["illegal_arrivals"] += 1
                job = Job(self.metrics["total_arrivals"], arrival_time, None, is_legal)
                self.mitigation_manager.handle_job(job)

    # -----------------------------------------------------------------
    # VALIDATION CSV: Web + Spike-0 (PRIMO SERVER) + Analysis ML
    # -----------------------------------------------------------------
    def export_validation_row(self, scenario: str, out_csv_path: str):
        now = self.env.now

        self.web_server.update(now)
        for s in self.spike_servers:
            s.update(now)
        center = self.mitigation_manager.center
        center.update(now)
        ac = self.mitigation_manager.analysis_center
        if ac is not None:
            ac.update(now)

        # ------ WEB (point)
        web_util_point = self._single_util(self.web_server, now)
        web_rt_mean_point = float(np.mean(self.web_server.completed_jobs)) if self.web_server.completed_jobs else 0.0
        web_thr_point = self.web_server.total_completions / max(now, 1e-12)

        # ------ SPIKE-0 (point) -> SOLO PRIMO SPIKE SERVER
        n_spk = len(self.spike_servers)
        s0 = self.spike_servers[0] if n_spk > 0 else None
        if s0 is not None:
            spike0_util_point = self._single_util(s0, now)
            spike0_rt_mean_point = float(np.mean(s0.completed_jobs)) if s0.completed_jobs else 0.0
            spike0_thr_point = s0.total_completions / max(now, 1e-12)
        else:
            spike0_util_point = 0.0
            spike0_rt_mean_point = 0.0
            spike0_thr_point = 0.0

        # ------ MITIGATION (point)
        mit_util_point = self._single_util(center, now)
        mit_rt_mean_point = float(np.mean(center.completed_jobs)) if center.completed_jobs else 0.0
        mit_thr_point = center.total_completions / max(now, 1e-12)

        # ------ ANALYSIS (point)
        if ac is not None:
            ana_util_point = self._analysis_util_rate_based(ac, now)
            ana_rt_mean_point = float(np.mean(ac.completed_jobs)) if ac.completed_jobs else 0.0
            ana_thr_point = ac.total_completions / max(now, 1e-12)
        else:
            ana_util_point = None
            ana_rt_mean_point = None
            ana_thr_point = None

        # ------ Drop rates
        drop_fp_rate = self.metrics.get("false_positives", 0) / max(now, 1e-12)
        drop_full_rate = (
            self.metrics.get("discarded_mitigation", 0)
            + len(self.metrics.get("discarded_detail", []))
            + self.metrics.get("discarded_analysis_capacity", 0)
        ) / max(now, 1e-12)

        # ================= Batch Means / Windowing =================
        web_util_series, web_thr_series = window_util_thr(
            self.web_server.busy_periods, self.web_server.completion_times, TIME_WINDOW, now
        )
        web_rt_bm_mean, web_rt_bm_ci = batch_means(self.web_server.completed_jobs, BATCH_SIZE)

        # ---- SPIKE-0 (BM) -> SOLO PRIMO SPIKE SERVER
        if s0 is not None:
            spike0_util_series, spike0_thr_series = window_util_thr(
                s0.busy_periods, s0.completion_times, TIME_WINDOW, now
            )
            if s0.completed_jobs and len(s0.completed_jobs) >= BATCH_SIZE * 2:
                spike0_rt_bm_mean, spike0_rt_bm_ci = batch_means(s0.completed_jobs, BATCH_SIZE)
            else:
                spike0_rt_bm_mean = float(np.mean(s0.completed_jobs)) if s0.completed_jobs else 0.0
                spike0_rt_bm_ci = 0.0
        else:
            spike0_util_series, spike0_thr_series = [], []
            spike0_rt_bm_mean, spike0_rt_bm_ci = 0.0, 0.0

        mit_util_series, mit_thr_series = window_util_thr(
            center.busy_periods, center.completion_times, TIME_WINDOW, now
        )
        mit_rt_bm_mean, mit_rt_bm_ci = batch_means(center.completed_jobs, BATCH_SIZE)

        if ac is not None:
            ana_util_series, ana_thr_series = self._window_util_thr_analysis_rate_based(ac, TIME_WINDOW, now)
            if ac.completed_jobs and len(ac.completed_jobs) >= BATCH_SIZE * 2:
                ana_rt_bm_mean, ana_rt_bm_ci = batch_means(ac.completed_jobs, BATCH_SIZE)
            else:
                ana_rt_bm_mean = float(np.mean(ac.completed_jobs)) if ac.completed_jobs else 0.0
                ana_rt_bm_ci = 0.0
        else:
            ana_util_series, ana_thr_series = [], []
            ana_rt_bm_mean, ana_rt_bm_ci = None, None

        def bm_mean_ci(series, win_per_batch):
            if series and len(series) // win_per_batch >= 2:
                m, hw = batch_means(series, win_per_batch)
                return float(m), float(hw)
            return (float(np.mean(series)) if series else 0.0, 0.0)

        web_util_bm_mean, web_util_bm_ci = bm_mean_ci(web_util_series, TIME_WINDOWS_PER_BATCH)
        web_thr_bm_mean,  web_thr_bm_ci  = bm_mean_ci(web_thr_series,  TIME_WINDOWS_PER_BATCH)

        spike0_util_bm_mean, spike0_util_bm_ci = bm_mean_ci(spike0_util_series, TIME_WINDOWS_PER_BATCH)
        spike0_thr_bm_mean,  spike0_thr_bm_ci  = bm_mean_ci(spike0_thr_series,  TIME_WINDOWS_PER_BATCH)

        mit_util_bm_mean, mit_util_bm_ci = bm_mean_ci(mit_util_series, TIME_WINDOWS_PER_BATCH)
        mit_thr_bm_mean,  mit_thr_bm_ci  = bm_mean_ci(mit_thr_series,  TIME_WINDOWS_PER_BATCH)

        if ac is not None:
            ana_util_bm_mean, ana_util_bm_ci = bm_mean_ci(ana_util_series, TIME_WINDOWS_PER_BATCH)
            ana_thr_bm_mean,  ana_thr_bm_ci  = bm_mean_ci(ana_thr_series,  TIME_WINDOWS_PER_BATCH)
        else:
            ana_util_bm_mean = ana_util_bm_ci = None
            ana_thr_bm_mean  = ana_thr_bm_ci  = None

        row = {
            "scenario": scenario,
            "ARRIVAL_P": self.arrival_p,
            "ARRIVAL_L1": self.arrival_l1,
            "ARRIVAL_L2": self.arrival_l2,

            "total_time": now,
            "total_arrivals": self.metrics.get("total_arrivals", 0),

            "web_util": web_util_point,
            "web_rt_mean": web_rt_mean_point,
            "web_throughput": web_thr_point,

            "spikes_count": n_spk,
            "spike0_util": spike0_util_point,
            "spike0_rt_mean": spike0_rt_mean_point,
            "spike0_throughput": spike0_thr_point,

            "mit_util": mit_util_point,
            "mit_rt_mean": mit_rt_mean_point,
            "mit_throughput": mit_thr_point,

            "drop_fp_rate": drop_fp_rate,
            "drop_full_rate": drop_full_rate,

            "web_util_bm_mean": web_util_bm_mean,
            "web_util_bm_ci": web_util_bm_ci,
            "web_thr_bm_mean": web_thr_bm_mean,
            "web_thr_bm_ci": web_thr_bm_ci,
            "web_rt_bm_mean": web_rt_bm_mean,
            "web_rt_bm_ci": web_rt_bm_ci,

            "spike0_util_bm_mean": spike0_util_bm_mean,
            "spike0_util_bm_ci": spike0_util_bm_ci,
            "spike0_thr_bm_mean": spike0_thr_bm_mean,
            "spike0_thr_bm_ci": spike0_thr_bm_ci,
            "spike0_rt_bm_mean": spike0_rt_bm_mean,
            "spike0_rt_bm_ci": spike0_rt_bm_ci,

            "mit_util_bm_mean": mit_util_bm_mean,
            "mit_util_bm_ci": mit_util_bm_ci,
            "mit_thr_bm_mean": mit_thr_bm_mean,
            "mit_thr_bm_ci": mit_thr_bm_ci,
            "mit_rt_bm_mean": mit_rt_bm_mean,
            "mit_rt_bm_ci": mit_rt_bm_ci,

            "analysis_util": ana_util_point,
            "analysis_rt_mean": ana_rt_mean_point,
            "analysis_throughput": ana_thr_point,

            "ana_util_bm_mean": ana_util_bm_mean,
            "ana_util_bm_ci": ana_util_bm_ci,
            "ana_thr_bm_mean": ana_thr_bm_mean,
            "ana_thr_bm_ci": ana_thr_bm_ci,
            "ana_rt_bm_mean": ana_rt_bm_mean,
            "ana_rt_bm_ci": ana_rt_bm_ci,

            "bm_rt_batch_size": BATCH_SIZE,
            "bm_win_size": TIME_WINDOW,
            "bm_windows_per_batch": TIME_WINDOWS_PER_BATCH
        }

        fns = validation_fieldnames()
        append_row_stable(out_csv_path, row, fns)

    # ---------------------- REPORT (windowing) ----------------------
    def report_windowing(self):
        now = self.env.now
        self.web_server.update(now)
        for s in self.spike_servers:
            s.update(now)
        center = self.mitigation_manager.center
        center.update(now)

        ac = self.mitigation_manager.analysis_center
        if ac is not None:
            ac.update(now)

        def close_open_busy_periods(periods, now_):
            if periods and periods[-1][1] is None:
                periods[-1][1] = now_
        close_open_busy_periods(self.web_server.busy_periods, now)
        for s in self.spike_servers:
            close_open_busy_periods(s.busy_periods, now)
        close_open_busy_periods(center.busy_periods, now)

        print("\n==== SIMULATION COMPLETE ====")
        print(f"Total time: {now:.4f}")
        print(f"Arrivals: {self.metrics['total_arrivals']}")
        print(f"Lecite: {self.metrics['legal_arrivals']}, Illecite: {self.metrics['illegal_arrivals']}")
        print(f"Mitigation completions: {self.metrics['mitigation_completions']}")
        print(f"False positives (dropped): {self.metrics['false_positives']}")
        print(f"  Di cui lecite: {self.metrics['false_positives_legal']}")

        discarded = self.metrics.get("discarded_detail", [])
        print(f"Total discarded jobs: {len(discarded)}")

        def stats(name, server):
            if getattr(server, "completed_jobs", None) and server.completed_jobs:
                avg_rt = np.mean(server.completed_jobs)
                if ac is not None and server is ac:
                    util = self._analysis_util_rate_based(ac, now)
                else:
                    util = self._single_util(server, now)
                print(f"{name} Completions: {server.total_completions}")
                if hasattr(server, "legal_completions"):
                    print(f"  Lecite  : {server.legal_completions}")
                    print(f"  Illecite: {server.illegal_completions}")
                print(f"{name} Avg Resp Time: {avg_rt:.6f}")
                print(f"{name} Utilization: {util:.6f}")
                print(f"{name} Throughput: {server.total_completions / now:.6f}")
            else:
                print(f"{name} Completions: 0")

        stats("Web", self.web_server)
        for i, server in enumerate(self.spike_servers):
            stats(f"Spike-{i}", server)
        stats("Mitigation", self.mitigation_manager.center)
        print(f"Mitigation Discarded : {self.metrics.get('discarded_mitigation', 0)}")

        if ac is not None:
            if getattr(ac, "completed_jobs", None) and ac.completed_jobs:
                avg_rt_ac = np.mean(ac.completed_jobs)
                util_ac = self._analysis_util_rate_based(ac, now)
                thr_ac = _throughput_upto(ac.completion_times, now)
                print(f"Analysis Completions: {ac.total_completions}")
                if hasattr(ac, "legal_completions"):
                    print(f"  Lecite  : {ac.legal_completions}")
                    print(f"  Illecite: {ac.illegal_completions}")
                print(f"Analysis Avg Resp Time: {avg_rt_ac:.6f}")
                print(f"Analysis Utilization:  {util_ac:.6f}")
                print(f"Analysis Throughput:   {thr_ac:.6f}")
            else:
                print("Analysis Completions: 0")

        print("\n======== INTERVALLI DI CONFIDENZA ========")
        print("\n-- Web Server --")

        def print_ci(label, data, batch_size):
            try:
                if len(data) < batch_size or len(data) // batch_size < 2:
                    raise ValueError("Solo 0 campioni → insufficienti per almeno 2 batch.")
                mean, ci = batch_means(data, batch_size)
                print(f"{label}: {mean:.6f} ± {ci:.6f} (95% CI)")
            except Exception as e:
                print(f"{label}: errore - {e}")

        print_ci("Response Time:                           ", self.web_server.completed_jobs, BATCH_SIZE)
        util_samples, thr_samples = window_util_thr(
            self.web_server.busy_periods, self.web_server.completion_times, TIME_WINDOW, now
        )
        print_ci("Utilization:                             ", util_samples, TIME_WINDOWS_PER_BATCH)
        print_ci("Throughput:                              ", thr_samples, TIME_WINDOWS_PER_BATCH)

        print("\n-- Spike Server --")
        for i, server in enumerate(self.spike_servers):
            print(f"Spike-{i}:")
            print_ci("Response Time:                           ", server.completed_jobs, BATCH_SIZE)
            util_samples, thr_samples = window_util_thr(
                server.busy_periods, server.completion_times, TIME_WINDOW, now
            )
            print_ci("Utilization:                             ", util_samples, TIME_WINDOWS_PER_BATCH)
            print_ci("Throughput:                              ", thr_samples, TIME_WINDOWS_PER_BATCH)

        print("\n-- Mitigation Center --")
        center = self.mitigation_manager.center
        print_ci("Response Time:                           ", center.completed_jobs, BATCH_SIZE)
        util_samples, thr_samples = window_util_thr(
            center.busy_periods, center.completion_times, TIME_WINDOW, now
        )
        print_ci("Utilization:                             ", util_samples, TIME_WINDOWS_PER_BATCH)
        print_ci("Throughput:                              ", thr_samples, TIME_WINDOWS_PER_BATCH)

        print("\n-- Analysis Center --")
        if ac is not None:
            print_ci("Response Time:                           ", ac.completed_jobs, BATCH_SIZE)
            util_samples_ac, thr_samples_ac = self._window_util_thr_analysis_rate_based(
                ac, TIME_WINDOW, now
            )
            print_ci("Utilization:                             ", util_samples_ac, TIME_WINDOWS_PER_BATCH)
            print_ci("Throughput:                              ", thr_samples_ac, TIME_WINDOWS_PER_BATCH)
        else:
            print("N/A (modello baseline)")

        print("\n==== END OF REPORT ====")

    # ---------------------- REPORT (batch means classico) ----------------------
    def report_bm(self,
                  B: int = None,
                  K: int = None,
                  confidence: float = CONFIDENCE_LEVEL,
                  burn_in_rt: int = 0):
        B = B or BATCH_SIZE
        K = K or N_BATCH
        now = self.env.now

        def _close_open_busy_periods(periods, now_):
            if periods and periods[-1][1] is None:
                periods[-1][1] = now_
        self.web_server.update(now)
        for s in self.spike_servers:
            s.update(now)
        center = self.mitigation_manager.center
        center.update(now)
        ac = self.mitigation_manager.analysis_center
        if ac is not None:
            ac.update(now)

        _close_open_busy_periods(self.web_server.busy_periods, now)
        for s in self.spike_servers:
            _close_open_busy_periods(s.busy_periods, now)
        _close_open_busy_periods(center.busy_periods, now)

        def _print_ci(label, samples, *, batch_size, n_batches, conf, burn_in=0):
            try:
                mean, hw = batch_means(
                    samples,
                    batch_size=batch_size,
                    n_batches=n_batches,
                    confidence=conf,
                    burn_in=burn_in
                )
                print(f"{label}: {mean:.6f} ± {hw:.6f} ({int(conf*100)}% CI)")
            except Exception as e:
                print(f"{label}: errore - {e}")

        print("\n==== SIMULATION COMPLETE ====")
        print(f"Total time: {now:.4f}")
        print(f"Arrivals: {self.metrics['total_arrivals']}")
        print(f"Lecite: {self.metrics['legal_arrivals']}, Illecite: {self.metrics['illegal_arrivals']}")
        print(f"Mitigation completions: {self.metrics['mitigation_completions']}")
        print(f"False positives (dropped): {self.metrics['false_positives']}")
        print(f"  Di cui lecite: {self.metrics['false_positives_legal']}")
        print(f"Total discarded jobs: {len(self.metrics.get('discarded_detail', []))}")

        print("\n======== INTERVALLI DI CONFIDENZA (Batch Means sui RT) ========")

        _print_ci("Web RT",
                  self.web_server.completed_jobs,
                  batch_size=B, n_batches=K, conf=confidence, burn_in=burn_in_rt)

        for i, srv in enumerate(self.spike_servers):
            _print_ci(f"Spike-{i} RT",
                      srv.completed_jobs,
                      batch_size=B, n_batches=K, conf=confidence, burn_in=burn_in_rt)

        _print_ci("Mitigation RT",
                  center.completed_jobs,
                  batch_size=B, n_batches=K, conf=confidence, burn_in=burn_in_rt)

        if ac is not None:
            _print_ci("Analysis RT",
                      ac.completed_jobs,
                      batch_size=B, n_batches=K, conf=confidence, burn_in=burn_in_rt)

        print("\n======== INTERVALLI DI CONFIDENZA (Utilization / Throughput per batch) ========")

        def _print_util_thr_ci_for(label_prefix, busy_periods, completion_times):
            util_series, thr_series = util_thr_per_batch(
                busy_periods,
                completion_times,
                B=B,
                burn_in=burn_in_rt,
                k_max=None,
                tmax=now
            )
            _print_ci(f"{label_prefix} Utilization",
                      util_series, batch_size=1, n_batches=None, conf=confidence)
            _print_ci(f"{label_prefix} Throughput",
                      thr_series, batch_size=1, n_batches=None, conf=confidence)

        _print_util_thr_ci_for("Web",
                               self.web_server.busy_periods,
                               self.web_server.completion_times)

        for i, srv in enumerate(self.spike_servers):
            _print_util_thr_ci_for(f"Spike-{i}",
                                   srv.busy_periods,
                                   srv.completion_times)

        _print_util_thr_ci_for("Mitigation",
                               center.busy_periods,
                               center.completion_times)

        if ac is not None:
            # >>> CORRETTO: Analysis rate-based su finestra temporale
            util_series_ac, thr_series_ac = self._window_util_thr_analysis_rate_based(
                ac, TIME_WINDOW, now
            )
            _print_ci("Analysis Utilization",
                      util_series_ac, batch_size=1, n_batches=None, conf=confidence)
            _print_ci("Analysis Throughput",
                      thr_series_ac,  batch_size=1, n_batches=None, conf=confidence)

        print("\n==== END OF REPORT ====")

    # ---------------------- REPORT  ----------------------
    def report_single_run(self):
        now = self.env.now
        self.web_server.update(now)
        for s in self.spike_servers:
            s.update(now)
        center = self.mitigation_manager.center
        center.update(now)

        ac = self.mitigation_manager.analysis_center
        if ac is not None:
            ac.update(now)

        print("\n==== SIMULATION COMPLETE ====")
        print(f"Total time: {now:.4f}")
        print(f"Arrivals: {self.metrics['total_arrivals']}")
        print(f"Lecite: {self.metrics['legal_arrivals']}, Illecite: {self.metrics['illegal_arrivals']}")
        print(f"Mitigation completions: {self.metrics['mitigation_completions']}")
        print(f"False positives (dropped): {self.metrics['false_positives']}")
        print(f"  Di cui lecite: {self.metrics['false_positives_legal']}")

        discarded = self.metrics.get("discarded_detail", [])
        print(f"Total discarded jobs: {len(discarded)}")

        def stats(name, server):
            if getattr(server, "completed_jobs", None) and server.completed_jobs:
                avg_rt = np.mean(server.completed_jobs)
                ac_local = self.mitigation_manager.analysis_center
                if ac_local is not None and server is ac_local:
                    util = self._analysis_util_rate_based(ac_local, now)
                else:
                    util = self._single_util(server, now)
                print(f"{name} Completions: {server.total_completions}")
                if hasattr(server, "legal_completions"):
                    print(f"  Lecite  : {server.legal_completions}")
                    print(f"  Illecite: {server.illegal_completions}")
                print(f"{name} Avg Resp Time: {avg_rt:.6f}")
                print(f"{name} Utilization: {util:.6f}")
                print(f"{name} Throughput: {server.total_completions / now:.6f}")
            else:
                print(f"{name} Completions: 0")

        stats("Web", self.web_server)
        for i, server in enumerate(self.spike_servers):
            stats(f"Spike-{i}", server)
        stats("Mitigation", center)

        ac = self.mitigation_manager.analysis_center
        if ac is not None:
            stats("Analysis", ac)
            print(f"Analysis capacity drops : {self.metrics.get('discarded_analysis_capacity', 0)}")
            print(f"ML  drop illegal (TN)   : {self.metrics.get('ml_drop_illegal', 0)}")
            print(f"ML  drop legal   (FN)   : {self.metrics.get('ml_drop_legal', 0)}")
            print(f"ML  pass illegal (FP)   : {self.metrics.get('ml_pass_illegal', 0)}")
            print(f"ML  pass legal   (TP)   : {self.metrics.get('ml_pass_legal', 0)}")

        print(f"Mitigation Discarded : {self.metrics.get('discarded_mitigation', 0)}")


# ---------------------------------------------------------------------
# Runner: single run / standard
# ---------------------------------------------------------------------
def run_simulation(scenario: str, mode: str, model: str, enable_windowing: bool,
                   arrival_p=None, arrival_l1=None, arrival_l2=None):

    print("Model " + model)

    if mode not in ("verification", "standard"):
        raise ValueError("mode must be 'verification' or 'standard'")
    if arrival_p is None:
        arrival_p = ARRIVAL_P
        arrival_l1 = ARRIVAL_L1
        arrival_l2 = ARRIVAL_L2

    env = simpy.Environment()
    system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2, variant=model)
    env.run()

    system.report_single_run()
    if enable_windowing:
        system.report_windowing()

    if scenario.lower() in {"x1", "x2", "x5", "x10", "x40"}:
        out_csv = "plot/results_validation_" + model + ".csv"
        system.export_validation_row(scenario=scenario, out_csv_path=out_csv)
        print(f"[OK] Riga di validazione salvata in: {out_csv}")

# ---------------------------------------------------------------------
# Runner: verifica (exp distribuzioni)
# ---------------------------------------------------------------------
def run_verification(model: str,
                     enable_windowing: bool = True,
                     arrival_p: float = None,
                     arrival_l1: float = None,
                     arrival_l2: float = None):
    print("Model " + model)

    if arrival_p is None:
        arrival_p = ARRIVAL_P_VERIFICATION
        arrival_l1 = ARRIVAL_L1_VERIFICATION
        arrival_l2 = ARRIVAL_L2_VERIFICATION

    env = simpy.Environment()
    system = DDoSSystem(env, mode="verification",
                        arrival_p=arrival_p, arrival_l1=arrival_l1, arrival_l2=arrival_l2,
                        variant=model)
    env.run()

    system.report_single_run()
    if enable_windowing:
        system.report_windowing()




# ---------------------------------------------------------------------
# Runner: orizzonte finito / transitorio
# ---------------------------------------------------------------------
def run_finite_horizon(mode: str, scenario: str, out_csv: str, model: str = "baseline"):
    if os.path.exists(out_csv):
        os.remove(out_csv)

    arrival_p = ARRIVAL_P
    arrival_l1 = ARRIVAL_L1_x40
    arrival_l2 = ARRIVAL_L2_x40

    fieldnames = transitory_fieldnames(MAX_SPIKE_NUMBER)
    all_logs = []

    if mode == "transitory":
        for rep in range(REPLICATION_FACTOR_TRANSITORY):
            seed = SEEDS_TRANSITORY[rep % len(SEEDS_TRANSITORY)]
            plantSeeds(seed)

            env = simpy.Environment()
            system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2, variant=model)

            def checkpointer_optimized(env, system, rep_id):
                checkpoint_times = []
                t = CHECKPOINT_TIME_TRANSITORY
                while t <= STOP_CONDITION_TRANSITORY:
                    checkpoint_times.append(t)
                    t += CHECKPOINT_TIME_TRANSITORY

                for cp_time in checkpoint_times:
                    yield env.timeout(cp_time - env.now)
                    snap = system.snapshot(env.now, replica_id=rep_id)
                    snap["scenario"] = scenario
                    snap["mode"] = mode
                    snap["is_final"] = (cp_time == checkpoint_times[-1])
                    all_logs.append(snap)
                    append_row_stable(out_csv, snap, fieldnames)
                    progress = (cp_time / STOP_CONDITION_TRANSITORY) * 100
                    print(f"  Replica {rep+1}: {progress:.1f}% completato")

            env.process(checkpointer_optimized(env, system, rep))
            env.run(until=STOP_CONDITION_TRANSITORY)
            print(f"Replica {rep+1} completata!")

    elif mode == "finite simulation":
        seeds = [1234566789]
        for rep in range(REPLICATION_FACTORY_FINITE_SIMULATION):
            plantSeeds(seeds[rep])
            env = simpy.Environment()
            system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2, variant=model)

            def checkpointer(env, system, rep_id):
                next_cp = CHECKPOINT_TIME_FINITE_SIMULATION
                while next_cp <= STOP_CONDITION_FINITE_SIMULATION:
                    yield env.timeout(next_cp - env.now)
                    snap = system.snapshot(env.now, replica_id=rep_id)
                    snap["scenario"] = scenario
                    snap["mode"] = mode
                    snap["is_final"] = False
                    all_logs.append(snap)
                    append_row_stable(out_csv, snap, fieldnames)
                    next_cp += CHECKPOINT_TIME_FINITE_SIMULATION

                if env.now < STOP_CONDITION_FINITE_SIMULATION:
                    yield env.timeout(STOP_CONDITION_FINITE_SIMULATION - env.now)
                    snap = system.snapshot(env.now, replica_id=rep_id)
                    snap["scenario"] = scenario
                    snap["mode"] = mode
                    snap["is_final"] = True
                    all_logs.append(snap)
                    append_row_stable(out_csv, snap, fieldnames)

            env.process(checkpointer(env, system, rep))
            env.run(until=STOP_CONDITION_FINITE_SIMULATION)

            selectStream(RNG_STREAM)
            seeds.append(getSeed())

    return all_logs


# ---------------------------------------------------------------------
# Runner: orizzonte infinito (Analysis + fix CSV wide ≥64 righe)
# ---------------------------------------------------------------------
def run_infinite_horizon(mode: str,
                         out_csv: str,
                         out_acs: str,
                         burn_in: int = 0,
                         arrival_p=None, arrival_l1=None, arrival_l2=None,
                         model: str = "baseline"):
    """
    Esegue la simulazione a orizzonte infinito tramite il metodo dei Batch Means,
    calcolando le autocorrelazioni (ACF) sulle metriche di interesse.

    - Aggiunge al CSV wide anche le serie per-batch del centro Analysis (se presente):
      'ana_rt', 'ana_util', 'ana_thr' (padding con NaN al bisogno).
    - Usa K dinamico per l'ACF: K = min(50, n-1) dove n è la lunghezza utile (non-NaN)
      più piccola tra le colonne selezionate, così si evita l'errore “length of data must be greater than K”.
    - Normalizza il CSV wide per i grafici: 21 colonne richieste e ≥64 righe.
    """
    if mode not in ("verification", "standard"):
        raise ValueError("mode must be 'verification' or 'standard'")

    if arrival_p is None:
        arrival_p = ARRIVAL_P
        arrival_l1 = ARRIVAL_L1_x40
        arrival_l2 = ARRIVAL_L2_x40

    env = simpy.Environment()
    system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2, variant=model)
    env.run()

    print("\n==== START BATCH MEANS ====")
    csv_path, cols, k = export_bm_series_to_wide_csv(
        system,
        B=BATCH_SIZE,
        out_csv=out_csv,
        burn_in=burn_in,
        k_max=None
    )
    print(f"[OK] bm_series in {csv_path} ({k} righe). Colonne: {cols}")

    # ---- Serie Analysis: append al CSV (se il centro esiste) ----
    ac = system.mitigation_manager.analysis_center
    if ac is not None:
        now = system.env.now
        ac.update(now)

        # >>> CORRETTO: serie Analysis rate-based su finestra temporale
        ana_util_series, ana_thr_series = system._window_util_thr_analysis_rate_based(
            ac, TIME_WINDOW, now
        )

        # Serie per-batch: RT (media su chunk consecutivi di B completamenti)
        rts = list(ac.completed_jobs)[burn_in:]
        n_batches_rt = len(rts) // BATCH_SIZE
        rt_series_ac = []
        for i in range(n_batches_rt):
            chunk = rts[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            if chunk:
                rt_series_ac.append(float(np.mean(chunk)))

        # Append/padding in CSV wide
        df = pd.read_csv(csv_path)

        def _pad(series, L):
            s = list(series)
            if len(s) < L:
                s = s + [np.nan] * (L - len(s))
            return s[:L]

        L = len(df)
        df["ana_rt"]   = _pad(rt_series_ac,   L)
        df["ana_util"] = _pad(ana_util_series, L)
        df["ana_thr"]  = _pad(ana_thr_series,  L)
        df.to_csv(csv_path, index=False)

        cols = [c for c in cols] + ["ana_rt", "ana_util", "ana_thr"]

    # 👉 FIX per grafici: colonne richieste e almeno 64 righe
    _fix_wide_csv_on_disk(csv_path, min_rows=64)

    # ---- Calcolo ACF con K dinamico (un unico K valido per tutte le colonne selezionate) ----
    df = pd.read_csv(csv_path)

    # Considera solo colonne presenti con almeno 2 osservazioni
    cols = [c for c in cols if c in df.columns and df[c].notna().sum() >= 2]
    if not cols:
        print("[WARN] Nessuna colonna con almeno 2 punti per ACF. Salto il calcolo.")
        return None

    n_per_col = {c: int(df[c].notna().sum()) for c in cols}
    min_n = min(n_per_col.values())
    K_dyn = max(1, min(50, min_n - 1))

    out_acs_clean = out_acs.replace(" ", "_")

    try:
        res_df = print_autocorrelation(
            file_path=csv_path,
            columns=cols,
            K_LAG=K_dyn,          
            threshold=0.2,
            save_csv=out_acs_clean
        )
        print(f"[OK] ACF salvata in: {out_acs_clean}")
        print(f"[info] ACF K dinamico = {K_dyn} (min punti tra {cols} = {min_n})")
        return res_df
    except Exception as e:
        print(f"[ERROR] ACF fallita: {e}")
        return None
