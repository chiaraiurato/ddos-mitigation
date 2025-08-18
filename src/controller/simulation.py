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
from engineering.statistics import batch_means, batch_means_cut_burn_in, make_batch_means_series
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
    def __init__(self, env, mode, arrival_p, arrival_l1, arrival_l2):
        self.env = env
        self.mode = mode
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
    def report_windowing(self):
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

        # CI con finestre per finito/se richiesto
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

        
    
    def report_bm(self):
        now = self.env.now
        self.web_server.update(now)
        for s in self.spike_servers: s.update(now)
        center = self.mitigation_manager.center
        center.update(now)

        def close_open_busy_periods(periods, now_):
            if periods and periods[-1][1] is None:
                periods[-1][1] = now_
        def _print_bm_ci(label, series, batch_size, n_batches, confidence=CONFIDENCE_LEVEL, burn_in=0):
            try:
                m, hw = batch_means_cut_burn_in(series, batch_size=batch_size, n_batches=n_batches,
                                    confidence=confidence, burn_in=burn_in)
                print(f"{label}: {m:.6f} ± {hw:.6f} ({int(confidence*100)}% CI)")
            except Exception as e:
                print(f"{label}: errore batch means - {e}")

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

        
        print("\n======== INTERVALLI DI CONFIDENZA (Batch Means sui RT) ========")
        burn_in_rt = 0  # per la run finita; per l’orizzonte infinito useremo un parametro dedicato
        # Web
        _print_bm_ci("Web RT", self.web_server.completed_jobs, BATCH_SIZE, N_BATCH, CONFIDENCE_LEVEL, burn_in=burn_in_rt)

        # Spike-i
        for i, server in enumerate(self.spike_servers):
            _print_bm_ci(f"Spike-{i} RT", server.completed_jobs, BATCH_SIZE, N_BATCH, CONFIDENCE_LEVEL, burn_in=burn_in_rt)

        # Mitigation
        _print_bm_ci("Mitigation RT", center.completed_jobs, BATCH_SIZE, N_BATCH, CONFIDENCE_LEVEL, burn_in=burn_in_rt)

        # System (Web+Spike)
        sys_rts = list(self.web_server.completed_jobs)
        for s in self.spike_servers:
            sys_rts.extend(s.completed_jobs)
        _print_bm_ci("System RT", sys_rts, BATCH_SIZE, N_BATCH, CONFIDENCE_LEVEL, burn_in=burn_in_rt)

        print("\n==== END OF REPORT ====")

def export_bm_series_for_acs(system,
                             B: int,
                             out_dir: str = "acs_input",
                             burn_in_rt: int = 0,
                             include_per_job: bool = False,
                             k_max=None):
    """
    Scrive file .dat con la SERIE delle medie di batch per le metriche RT:
      - web_rt_B<B>.dat
      - mit_rt_B<B>.dat
      - system_rt_B<B>.dat
      - spike<i>_rt_B<B>.dat per ogni spike presente
    Opzionale: scrive anche le serie per-job:
      - web_rt_perjob.dat, mit_rt_perjob.dat, system_rt_perjob.dat, spike<i>_rt_perjob.dat
    Ogni riga del file è un singolo valore (come vuole acs.py).
    """
    os.makedirs(out_dir, exist_ok=True)

    center = system.mitigation_manager.center

    # Serie per-job
    web_rt   = list(system.web_server.completed_jobs)
    mit_rt   = list(center.completed_jobs)
    spike_rts = [list(s.completed_jobs) for s in system.spike_servers]
    sys_rt = list(web_rt)
    for s in system.spike_servers:
        sys_rt.extend(s.completed_jobs)

    items = [("web_rt", web_rt),
             ("mit_rt", mit_rt),
             ("system_rt", sys_rt)]
    for i, r in enumerate(spike_rts):
        items.append((f"spike{i}_rt", r))

    def _dump(path, seq):
        with open(path, "w") as f:
            for v in seq:
                f.write(f"{float(v)}\n")

    for name, series in items:
        # opzionale: per-job
        if include_per_job:
            _dump(os.path.join(out_dir, f"{name}_perjob.dat"), series)

        # batch-means per B scelto
        means = make_batch_means_series(series, b=B, burn_in=burn_in_rt, k_max=k_max)
        _dump(os.path.join(out_dir, f"{name}_B{B}.dat"), means)

    print(f"[OK] Esportati i file .dat in '{out_dir}' per B={B} (burn-in={burn_in_rt}, k_max={k_max})")


def run_simulation(scenario:str, mode: str, enable_windowing: bool, arrival_p=None, arrival_l1=None, arrival_l2=None):
    if mode not in ("verification", "standard"):
        raise ValueError("mode must be 'verification' or 'standard'")
    if arrival_p is None:
        arrival_p = ARRIVAL_P
        arrival_l1 = ARRIVAL_L1
        arrival_l2 = ARRIVAL_L2

    env = simpy.Environment()
    system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2)
    env.run()
    if(enable_windowing == True):
        system.report_windowing()

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

def run_finite_sim(mode: str, enable_batch_means: bool, scenario: str, out_csv: str):
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
            system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2)

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
            system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2)

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

def run_infinite_horizon_bm_to_csv(scenario: str,
                                   mode: str,
                                   out_csv: str,
                                   burn_in_rt: int,
                                   arrival_p=None, arrival_l1=None, arrival_l2=None):
    """
    Una singola run 'infinita' che produce un'unica riga CSV:
    - RT: Batch Means con CI (no finestre)
    - Util/Throughput: stime puntuali (totBusy/now, completamenti/now)
    """
    if mode not in ("verification", "standard"):
        raise ValueError("mode must be 'verification' or 'standard'")

    if arrival_p is None:
        arrival_p  = ARRIVAL_P
        arrival_l1 = ARRIVAL_L1
        arrival_l2 = ARRIVAL_L2

    # pulizia file
    if os.path.exists(out_csv):
        os.remove(out_csv)

    env = simpy.Environment()
    system = DDoSSystem(env, mode, arrival_p, arrival_l1, arrival_l2)
    env.run()
    export_bm_series_for_acs(
    system,
    B=BATCH_SIZE,                 
    out_dir="acs_input",  
    burn_in_rt=5000,       
    include_per_job=False, 
    k_max=None             
    )
    now = env.now
    center = system.mitigation_manager.center

    
    need = burn_in_rt + BATCH_SIZE * N_BATCH
    def _check_len(name, seq):
        if len(seq) < need:
            raise ValueError(f"{name}: completamenti insufficienti per BM (len={len(seq)} < {need}). "
                             f"Aumenta N_ARRIVALS o riduci BATCH_SIZE/N_BATCH/burn_in_rt.")

   
    web_rt    = list(system.web_server.completed_jobs)
    mit_rt    = list(center.completed_jobs)
    spike_rts = [list(s.completed_jobs) for s in system.spike_servers]
    sys_rt    = list(web_rt)
    for s in system.spike_servers:
        sys_rt.extend(s.completed_jobs)

    # check quantità
    _check_len("Web RT", web_rt)
    _check_len("Mitigation RT", mit_rt)
    for i, r in enumerate(spike_rts):
        _check_len(f"Spike-{i} RT", r)
    _check_len("System RT", sys_rt)

    
    def _bm_rt(x): 
        return batch_means_cut_burn_in(x, batch_size=BATCH_SIZE, n_batches=N_BATCH,
                                       confidence=CONFIDENCE_LEVEL, burn_in=burn_in_rt)

    web_m, web_hw = _bm_rt(web_rt)
    mit_m, mit_hw = _bm_rt(mit_rt)
    sys_m, sys_hw = _bm_rt(sys_rt)
    spike_stats = [ _bm_rt(r) for r in spike_rts ]

    # --- stime puntuali util/thr
    def _util(srv): return (srv.busy_time / now) if now > 0 else 0.0
    def _thr(srv):  return (srv.total_completions / now) if now > 0 else 0.0

    web_util_p, web_thr_p = _util(system.web_server), _thr(system.web_server)
    mit_util_p, mit_thr_p = _util(center), _thr(center)

    # --- quote globali
    tot_arr = max(1, system.metrics["total_arrivals"])
    proc_leg = int(system.metrics.get("processed_legal", 0))
    proc_illeg = int(system.metrics.get("processed_illegal", 0))

    web_leg = int(system.web_server.legal_completions)
    web_illeg = int(system.web_server.illegal_completions)
    spike_leg = sum(int(s.legal_completions) for s in system.spike_servers)
    spike_illeg = sum(int(s.illegal_completions) for s in system.spike_servers)
    comp_leg = web_leg + spike_leg
    comp_illeg = web_illeg + spike_illeg
    comp_tot = max(1, comp_leg + comp_illeg)

    fieldnames = infinite_fieldnames(MAX_SPIKE_NUMBER)
    row = {
        "scenario": scenario,
        "total_time": now, "total_arrivals": system.metrics["total_arrivals"],
        "burn_in_rt": burn_in_rt, "batch_size": BATCH_SIZE, "n_batches": N_BATCH, "confidence": CONFIDENCE_LEVEL,

        "web_rt_mean_bm": web_m, "web_rt_ci_hw": web_hw,
        "web_util_point": web_util_p, "web_throughput_point": web_thr_p,

        "mit_rt_mean_bm": mit_m, "mit_rt_ci_hw": mit_hw,
        "mit_util_point": mit_util_p, "mit_throughput_point": mit_thr_p,

        "system_rt_mean_bm": sys_m, "system_rt_ci_hw": sys_hw,

        "illegal_share": system.metrics["illegal_arrivals"] / tot_arr,
        "processed_legal_share": proc_leg / tot_arr,
        "processed_illegal_share": proc_illeg / tot_arr,
        "completed_legal_share": comp_leg / tot_arr,
        "completed_illegal_share": comp_illeg / tot_arr,
        "completed_legal_of_completed_share": comp_leg / comp_tot,
        "completed_illegal_of_completed_share": comp_illeg / comp_tot,

        "spikes_count": len(system.spike_servers),
    }

    # spike-i
    for i in range(MAX_SPIKE_NUMBER):
        if i < len(system.spike_servers):
            s = system.spike_servers[i]
            m, hw = spike_stats[i]
            row[f"spike{i}_rt_mean_bm"] = m
            row[f"spike{i}_rt_ci_hw"]   = hw
            row[f"spike{i}_util_point"] = _util(s)
            row[f"spike{i}_thr_point"]  = _thr(s)
            row[f"spike{i}_completions"] = s.total_completions
        else:
            row[f"spike{i}_rt_mean_bm"] = None
            row[f"spike{i}_rt_ci_hw"]   = None
            row[f"spike{i}_util_point"] = None
            row[f"spike{i}_thr_point"]  = None
            row[f"spike{i}_completions"] = 0

    append_row_stable(out_csv, row, fieldnames)
    print(f"[OK] Batch means salvati in: {out_csv}")


def run_infinite_horizon(mode: str,
                         enable_batch_means: bool,
                         out_csv: str = "results_infinite_bm.csv",
                         burn_in_rt: int = 0,
                         arrival_p=None, arrival_l1=None, arrival_l2=None,
                         scenario: str = "infinite_run"):
    """
    Se enable_batch_means=True -> delega a run_infinite_horizon_bm_to_csv (CSV, no stampe).
    Altrimenti -> run classica e scrive la riga 'point-estimates' su results_standard.csv.
    """
    if enable_batch_means:
        return run_infinite_horizon_bm_to_csv(
            scenario=scenario, mode=mode, out_csv=out_csv, burn_in_rt=burn_in_rt,
            arrival_p=arrival_p, arrival_l1=arrival_l1, arrival_l2=arrival_l2
        )





