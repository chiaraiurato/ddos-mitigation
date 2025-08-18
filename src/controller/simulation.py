import simpy
import numpy as np
import os

from library.rngs import random, plantSeeds, getSeed, selectStream
from engineering.costants import *
from engineering.distributions import get_interarrival_time
from engineering.statistics import batch_means, window_util_thr
from utils.csv_writer import append_row, append_row_stable
from model.job import Job
from model.processor_sharing_server import ProcessorSharingServer
from model.mitigation_manager import MitigationManager
from utils.checkpoint import _rt_mean_upto, _utilization_upto, _throughput_upto

try:
    STOP_TRANSITORY = STOP_CONDITION_TRANSITORY
except NameError:
    try:
        STOP_TRANSITORY = STOP_CONDITION_TRANSITORY
    except NameError:
        STOP_TRANSITORY = STOP_CONDITION_FINITE_SIMULATION

# CHECKPOINT
try:
    CHECKPOINT_TRANSITORY = CHECKPOINT_TIME_TRANSITORY
except NameError:
    try:
        CHECKPOINT_TRANSITORY = CHECKPOINT_TIME_TRANSITORY
    except NameError:
        CHECKPOINT_TRANSITORY = CHECKPOINT_TIME_FINITE_SIMULATION


class DDoSSystem:
    def __init__(self, env, mode, arrival_p, arrival_l1, arrival_l2, enable_ci=False):
        self.env = env
        self.mode = mode
        self.enable_ci = enable_ci
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
            env, self.web_server, self.spike_servers, self.metrics, self.mode
        )

        self.env.process(self.arrival_process(mode))

    # ---------------------- SNAPSHOT (orizzonte finito) ----------------------
    def snapshot(self, when, replica_id=None):
        # flush stati al tempo 'when'
        self.web_server.update(when)
        for s in self.spike_servers:
            s.update(when)
        center = self.mitigation_manager.center
        center.update(when)

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

            # Contatori globali
            "arrivals_so_far": int(self.metrics["total_arrivals"]),
            "false_positives_so_far": int(self.metrics.get("false_positives", 0)),
            "mitigation_completions_so_far": int(self.metrics.get("mitigation_completions", 0)),
            "spikes_count": int(len(self.spike_servers)),
        }

        # Per-spike
        for i in range(len(self.spike_servers)):
            s = self.spike_servers[i]
            row[f"spike{i}_rt_mean"] = _rt_mean_upto(s.completed_jobs, s.completion_times, when)
            row[f"spike{i}_util"] = _utilization_upto(s.busy_periods, when)
            row[f"spike{i}_throughput"] = _throughput_upto(s.completion_times, when)

        # Padding colonne spike fino a MAX_SPIKE_NUMBER
        for i in range(len(self.spike_servers), MAX_SPIKE_NUMBER):
            row[f"spike{i}_rt_mean"] = None
            row[f"spike{i}_util"] = None
            row[f"spike{i}_throughput"] = None

        # Mix/quote fino a 'when'
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

        # RT medio globale (Web+Spike) fino a 'when'
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
        else:  # "standard"
            arrivals = N_ARRIVALS
            interarrival_mode = "standard"
            stop_on_arrivals = True

        if not stop_on_arrivals:
            while True:
                interarrival_time = get_interarrival_time(interarrival_mode, self.arrival_p, self.arrival_l1, self.arrival_l2)
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
                interarrival_time = get_interarrival_time(interarrival_mode, self.arrival_p, self.arrival_l1, self.arrival_l2)
                yield self.env.timeout(interarrival_time)
                self.metrics["total_arrivals"] += 1
                arrival_time = self.env.now
                is_legal = (random() < P_LECITO)
                if is_legal: self.metrics["legal_arrivals"] += 1
                else:        self.metrics["illegal_arrivals"] += 1
                job = Job(self.metrics["total_arrivals"], arrival_time, None, is_legal)
                self.mitigation_manager.handle_job(job)

    # ---------------------- REPORT (finito) ----------------------
    def report(self):
        now = self.env.now
        self.web_server.update(now)
        for s in self.spike_servers: s.update(now)
        center = self.mitigation_manager.center
        center.update(now)

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
                print(f"{name} Completions: {server.total_completions}")
                print(f"  Lecite  : {server.legal_completions}")
                print(f"  Illecite: {server.illegal_completions}")
                print(f"{name} Avg Resp Time: {avg_rt:.6f}")
                print(f"{name} Utilization: {server.busy_time / now:.6f}")
                print(f"{name} Throughput: {server.total_completions / now:.6f}")
            else:
                print(f"{name} Completions: 0")

        def print_ci(label, data, batch_size):
            try:
                if len(data) < batch_size or len(data) // batch_size < 2:
                    raise ValueError(f"Solo {len(data)} campioni → insufficienti per almeno 2 batch.")
                mean, ci = batch_means(data, batch_size)
                print(f"{label}: {mean:.6f} ± {ci:.6f} (95% CI)")
            except Exception as e:
                print(f"{label}: errore - {e}")

        stats("Web", self.web_server)
        for i, server in enumerate(self.spike_servers):
            stats(f"Spike-{i}", server)
        stats("Mitigation", center)
        print(f"Mitigation Discarded : {self.metrics.get('discarded_mitigation', 0)}")

        # CI (opzionale) con finestre per finito/se richiesto
        if self.enable_ci:
            print("\n======== INTERVALLI DI CONFIDENZA ========")
            print("\n-- Web Server --")
            print_ci("Response Time:                           ", self.web_server.completed_jobs, BATCH_SIZE)
            util_samples, thr_samples = window_util_thr(self.web_server.busy_periods, self.web_server.completion_times, TIME_WINDOW, now)
            print_ci("Utilization:                             ", util_samples, TIME_WINDOWS_PER_BATCH)
            print_ci("Throughput:                              ", thr_samples, TIME_WINDOWS_PER_BATCH)

            print("\n-- Spike Server --")
            for i, server in enumerate(self.spike_servers):
                print(f"Spike-{i}:")
                print_ci("Response Time:                           ", server.completed_jobs, BATCH_SIZE)
                util_samples, thr_samples = window_util_thr(server.busy_periods, server.completion_times, TIME_WINDOW, now)
                print_ci("Utilization:                             ", util_samples, TIME_WINDOWS_PER_BATCH)
                print_ci("Throughput:                              ", thr_samples, TIME_WINDOWS_PER_BATCH)

            print("\n-- Mitigation Center --")
            center = self.mitigation_manager.center
            print_ci("Response Time:                           ", center.completed_jobs, BATCH_SIZE)
            util_samples, thr_samples = window_util_thr(center.busy_periods, center.completion_times, TIME_WINDOW, now)
            print_ci("Utilization:                             ", util_samples, TIME_WINDOWS_PER_BATCH)
            print_ci("Throughput:                              ", thr_samples, TIME_WINDOWS_PER_BATCH)

        print("\n==== END OF REPORT ====")

    # ====== INFINITE-HORIZON (BM classico sugli RT, senza finestre) ======
    def _system_rt_sorted(self):
        """RT 'globali' (Web + Spike) ordinati per tempo di completamento."""
        pairs = []
        pairs.extend(zip(self.web_server.completion_times, self.web_server.completed_jobs))
        for s in self.spike_servers:
            pairs.extend(zip(s.completion_times, s.completed_jobs))
        pairs.sort(key=lambda x: x[0])
        return [rt for (_, rt) in pairs]

    @staticmethod
    def _busy_between(periods, t0, t1):
        busy = 0.0
        for (a, b) in periods:
            bb = t1 if b is None else b
            if bb <= t0 or a >= t1:
                continue
            busy += max(0.0, min(bb, t1) - max(a, t0))
        return busy

    def _point_estimates_since(self, server, t0):
        """Stime puntuali (util, thr) su [t0, now]."""
        now = self.env.now
        if now <= t0: return None, None
        busy = self._busy_between(server.busy_periods, t0, now)
        util = busy / (now - t0)
        comps = sum(1 for x in server.completion_times if x >= t0)
        thr = comps / (now - t0)
        return util, thr

    def _rt_ci(self, series, burn_in_rt):
        """BM classico sugli RT con fallback se dati insufficienti."""
        from engineering.statistics import batch_means_rt
        arr = list(series)
        if burn_in_rt and len(arr) > burn_in_rt:
            arr = arr[burn_in_rt:]
        try:
            return batch_means_rt(
                arr,
                batch_size=globals().get("BATCH_SIZE", None),
                n_batches=globals().get("N_BATCH", None),
                confidence=globals().get("CONFIDENCE_LEVEL", None),
                burn_in=0
            )
        except Exception:
            # fallback: media semplice, nessun half-width
            return (float(np.mean(arr)) if len(arr) > 0 else 0.0, "")

    def make_infinite_csv_row(self, scenario: str, arrival_p: float, arrival_l1: float,
                              arrival_l2: float, burn_in_rt: int):
        """Riga CSV: IC BM RT + stime puntuali util/thr + mix/quote."""
        now = self.env.now
        self.web_server.update(now)
        for s in self.spike_servers: s.update(now)
        center = self.mitigation_manager.center
        center.update(now)

        # ---- RT: IC BM classico ----
        web_rt_mean, web_rt_hw = self._rt_ci(self.web_server.completed_jobs, burn_in_rt)
        mit_rt_mean, mit_rt_hw = self._rt_ci(center.completed_jobs, burn_in_rt)

        sys_series = self._system_rt_sorted()
        sys_rt_mean, sys_rt_hw = self._rt_ci(sys_series, burn_in_rt)

        spike_ci = []
        for s in self.spike_servers:
            m, h = self._rt_ci(s.completed_jobs, burn_in_rt)
            spike_ci.append((m, h))

        # ---- Util & Thr: stime puntuali post-burn-in ----
        def t0_from(srv):
            ct = getattr(srv, "completion_times", [])
            if burn_in_rt > 0 and len(ct) >= burn_in_rt:
                return ct[burn_in_rt - 1]
            return 0.0

        t0_web = t0_from(self.web_server)
        t0_mit = t0_from(center)

        web_util, web_thr = self._point_estimates_since(self.web_server, t0_web)
        mit_util, mit_thr = self._point_estimates_since(center, t0_mit)

        spikes_util_thr = []
        for s in self.spike_servers:
            t0s = t0_from(s)
            u, t = self._point_estimates_since(s, t0s)
            spikes_util_thr.append((u, t))

        # ---- Quote lecite/illecite ----
        tot_arr = max(1, int(self.metrics.get("total_arrivals", 0)))
        proc_leg = int(self.metrics.get("processed_legal", 0))
        proc_illeg = int(self.metrics.get("processed_illegal", 0))

        web_leg = int(self.web_server.legal_completions)
        web_illeg = int(self.web_server.illegal_completions)
        spike_leg = sum(int(s.legal_completions) for s in self.spike_servers)
        spike_il = sum(int(s.illegal_completions) for s in self.spike_servers)

        comp_leg = web_leg + spike_leg
        comp_illeg = web_illeg + spike_il
        comp_tot = max(1, comp_leg + comp_illeg)

        row = {
            # meta & setup
            "scenario": scenario,
            "ARRIVAL_P": float(arrival_p),
            "ARRIVAL_L1": float(arrival_l1),
            "ARRIVAL_L2": float(arrival_l2),
            "total_time": float(now),
            "total_arrivals": int(self.metrics.get("total_arrivals", 0)),
            "burn_in_rt": int(burn_in_rt),
            "batch_size": int(globals().get("BATCH_SIZE", 0) or 0),
            "n_batches": int(globals().get("N_BATCH", 0) or 0),
            "confidence": float(globals().get("CONFIDENCE_LEVEL", 0.95)),

            # web
            "web_rt_mean_bm": float(web_rt_mean),
            "web_rt_ci_hw": web_rt_hw if web_rt_hw == "" else float(web_rt_hw),
            "web_util_point": float(web_util) if web_util is not None else "",
            "web_throughput_point": float(web_thr) if web_thr is not None else "",

            # mitigation
            "mit_rt_mean_bm": float(mit_rt_mean),
            "mit_rt_ci_hw": mit_rt_hw if mit_rt_hw == "" else float(mit_rt_hw),
            "mit_util_point": float(mit_util) if mit_util is not None else "",
            "mit_throughput_point": float(mit_thr) if mit_thr is not None else "",

            # globale (Web + Spike)
            "system_rt_mean_bm": float(sys_rt_mean),
            "system_rt_ci_hw": sys_rt_hw if sys_rt_hw == "" else float(sys_rt_hw),

            # quote
            "illegal_share": float(self.metrics.get("illegal_arrivals", 0)) / tot_arr,
            "processed_legal_share": float(proc_leg) / tot_arr,
            "processed_illegal_share": float(proc_illeg) / tot_arr,
            "completed_legal_share": float(comp_leg) / tot_arr,
            "completed_illegal_share": float(comp_illeg) / tot_arr,
            "completed_legal_of_completed_share": float(comp_leg) / comp_tot,
            "completed_illegal_of_completed_share": float(comp_illeg) / comp_tot,

            "spikes_count": int(len(self.spike_servers)),
        }

        # per-spike
        for i in range(MAX_SPIKE_NUMBER):
            if i < len(spike_ci):
                (m, h) = spike_ci[i]
                u, t = spikes_util_thr[i]
                row[f"spike{i}_rt_mean_bm"] = float(m)
                row[f"spike{i}_rt_ci_hw"] = h if h == "" else float(h)
                row[f"spike{i}_util_point"] = float(u) if u is not None else ""
                row[f"spike{i}_thr_point"] = float(t) if t is not None else ""
                row[f"spike{i}_completions"] = int(self.spike_servers[i].total_completions)
            else:
                row[f"spike{i}_rt_mean_bm"] = ""
                row[f"spike{i}_rt_ci_hw"] = ""
                row[f"spike{i}_util_point"] = ""
                row[f"spike{i}_thr_point"] = ""
                row[f"spike{i}_completions"] = ""
        return row

    def report_infinite_bm(self, burn_in_rt=0):
        """Stampa a video i risultati (opzionale, comodo in debug)."""
        now = self.env.now
        self.web_server.update(now)
        for s in self.spike_servers: s.update(now)
        self.mitigation_manager.center.update(now)

        from engineering.statistics import batch_means_rt
        print("\n==== INFINITE-HORIZON — Batch Means (RT) senza finestre ====")

        def safe_bm(label, series):
            try:
                m, h = batch_means_rt(series,
                                      batch_size=globals().get("BATCH_SIZE", None),
                                      n_batches=globals().get("N_BATCH", None),
                                      confidence=globals().get("CONFIDENCE_LEVEL", None),
                                      burn_in=burn_in_rt)
                print(f"{label}: {m:.6f} ± {h:.6f} (95% CI)")
            except Exception as e:
                if len(series) > 0:
                    print(f"{label}: {np.mean(series):.6f} (no CI: {e})")
                else:
                    print(f"{label}: n=0 (no data)")

        safe_bm("Web RT", self.web_server.completed_jobs)
        for i, s in enumerate(self.spike_servers):
            safe_bm(f"Spike-{i} RT", s.completed_jobs)
        safe_bm("Mitigation RT", self.mitigation_manager.center.completed_jobs)
        safe_bm("System RT", self._system_rt_sorted())


# -------------------------- API “finite” già esistenti --------------------------
def run_simulation(scenario:str, mode: str, batch_means: bool, arrival_p=None, arrival_l1=None, arrival_l2=None):
    if mode not in ("verification", "standard"):
        raise ValueError("mode must be 'verification' or 'standard'")
    if arrival_p is None:
        arrival_p = ARRIVAL_P
        arrival_l1 = ARRIVAL_L1
        arrival_l2 = ARRIVAL_L2

    env = simpy.Environment()
    system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2, enable_ci=batch_means)
    env.run()
    system.report()

    row = system.make_csv_row(scenario, arrival_p, arrival_l1, arrival_l2)
    append_row("results_standard.csv", row)

def transitory_fieldnames(max_spikes: int):
    base = [
        "replica", "time",
        "web_rt_mean", "web_util", "web_throughput",
        "mit_rt_mean", "mit_util", "mit_throughput",
        "system_rt_mean",
        "arrivals_so_far", "false_positives_so_far", "mitigation_completions_so_far",
        "spikes_count", "scenario", "mode", "is_final",
        "illegal_share",
        "processed_legal_share", "processed_illegal_share",
        "completed_legal_share", "completed_illegal_share",
        "completed_legal_of_completed_share", "completed_illegal_of_completed_share",
    ]
    for i in range(MAX_SPIKE_NUMBER):
        base += [f"spike{i}_rt_mean", f"spike{i}_util", f"spike{i}_throughput"]
    return base

def infinite_fieldnames(max_spikes: int):
    base = [
        "scenario",
        "ARRIVAL_P", "ARRIVAL_L1", "ARRIVAL_L2",
        "total_time", "total_arrivals",
        "burn_in_rt", "batch_size", "n_batches", "confidence",
        "web_rt_mean_bm", "web_rt_ci_hw",
        "web_util_point", "web_throughput_point",
        "mit_rt_mean_bm", "mit_rt_ci_hw",
        "mit_util_point", "mit_throughput_point",
        "system_rt_mean_bm", "system_rt_ci_hw",
        "illegal_share",
        "processed_legal_share", "processed_illegal_share",
        "completed_legal_share", "completed_illegal_share",
        "completed_legal_of_completed_share", "completed_illegal_of_completed_share",
        "spikes_count",
    ]
    for i in range(MAX_SPIKE_NUMBER):
        base += [f"spike{i}_rt_mean_bm", f"spike{i}_rt_ci_hw",
                 f"spike{i}_util_point", f"spike{i}_thr_point",
                 f"spike{i}_completions"]
    return base

def run_finite_sim(mode: str, batch_means: bool, scenario: str, out_csv: str):
    if os.path.exists(out_csv):
        os.remove(out_csv)

    arrival_p  = ARRIVAL_P
    arrival_l1 = ARRIVAL_L1_x40
    arrival_l2 = ARRIVAL_L2_x40

    fieldnames = transitory_fieldnames(MAX_SPIKE_NUMBER)
    all_logs = []

    if mode == "transitory":
        for rep in range(REPLICATION_FACTOR_TRANSITORY):
            seed = SEEDS_TRANSITORY[rep % len(SEEDS_TRANSITORY)]
            plantSeeds(seed)

            env = simpy.Environment()
            system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2, enable_ci=batch_means)

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
            system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2, enable_ci=batch_means)

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

# -------------------------- NUOVO: run infinito (BM 1024x256) --------------------------
def run_infinite_horizon_bm_to_csv(
    scenario: str,
    mode: str = "standard",
    out_csv: str = "results_infinite_bm.csv",
    burn_in_rt: int = 0,
    arrival_p=None, arrival_l1=None, arrival_l2=None
):
    """
    Esegue una run 'infinita' (senza finestre) e salva una riga in out_csv
    con IC BM (RT) + stime puntuali util/thr + quote lecite/illecite.
    Stampa anche una barra di avanzamento (arrivi e completamenti BM).
    """
    if arrival_p is None:
        arrival_p  = ARRIVAL_P
        arrival_l1 = ARRIVAL_L1_x40
        arrival_l2 = ARRIVAL_L2_x40

    # obiettivi per la progress bar
    goal_batches = burn_in_rt + int(BATCH_SIZE) * int(N_BATCH)  # completamenti necessari per BM
    if mode == "verification":
        goal_arrivals = int(N_ARRIVALS_VERIFICATION)
    else:
        goal_arrivals = int(N_ARRIVALS)

    # intervallo di stampa in TEMPO DI SIMULAZIONE (puoi ritoccare se vuoi)
    PROGRESS_EVERY_SIMSEC = 300.0  # stampa circa ogni 5 minuti simulati

    def progress_meter(env, system):
        last_pct_arr = -1
        last_pct_bm  = -1
        while True:
            yield env.timeout(PROGRESS_EVERY_SIMSEC)

            # arrivi
            arr = int(system.metrics.get("total_arrivals", 0))
            pct_arr = min(100.0, 100.0 * arr / max(1, goal_arrivals))

            # completamenti minimi per BM (lower bound: min web/mit/sys)
            web_c = system.web_server.total_completions
            mit_c = system.mitigation_manager.center.total_completions
            sys_c = web_c + sum(s.total_completions for s in system.spike_servers)
            min_c = min(web_c, mit_c, sys_c)
            pct_bm = min(100.0, 100.0 * min_c / max(1, goal_batches))

            # stampa solo quando avanzi almeno di ~1% su una delle due scale
            if int(pct_arr) != int(last_pct_arr) or int(pct_bm) != int(last_pct_bm):
                print(
                    f"[PROGRESS] t={env.now:.0f}s | "
                    f"Arrivi: {arr}/{goal_arrivals} ({pct_arr:.1f}%) | "
                    f"BM(min web/mit/sys): {min_c}/{goal_batches} ({pct_bm:.1f}%) | "
                    f"Spike attivi: {len(system.spike_servers)}"
                )
                last_pct_arr = pct_arr
                last_pct_bm  = pct_bm

            # esci se gli arrivi sono completi (env.run() finirà subito dopo)
            if arr >= goal_arrivals:
                break

    env = simpy.Environment()
    system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2, enable_ci=False)

    # avvia il misuratore di avanzamento in parallelo
    env.process(progress_meter(env, system))

    # esegui la simulazione "standard" (basata sugli arrivi) fino a esaurimento eventi
    env.run()

    # prepara riga CSV e scrivi
    row = system.make_infinite_csv_row(scenario, arrival_p, arrival_l1, arrival_l2, burn_in_rt)
    fieldnames = infinite_fieldnames(MAX_SPIKE_NUMBER)  # header stabile
    append_row_stable(out_csv, row, fieldnames)

    # (opzionale) stampa un riassunto BM a video
    system.report_infinite_bm(burn_in_rt=burn_in_rt)

    print(f"[OK] Riga BM salvata in: {os.path.abspath(out_csv)}")
    return row

